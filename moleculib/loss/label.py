class CrossEntropyLoss(LossFunction):

    def _cross_entropy_loss(self, logits, labels, mask=None):
        if mask is not None:
            logits = jnp.where(mask[..., None], logits, jnp.array([1] + [0] * 22))
        cross_entropy = -(labels * jax.nn.log_softmax(logits)).sum(-1)
        return cross_entropy.mean()

    def _call(
        self, rng_key, model_output: ModelOutput, ground: ProteinDatum
    ) -> Tuple[ModelOutput, jax.Array, Dict[str, float]]:
        res_logits = model_output.datum["residue_logits"]
        ground = ground[0]

        total_loss, metrics = 0.0, {}

        labels = ground.residue_token
        res_mask = ground.atom_mask[..., 1]

        labels = rearrange(labels, "... -> (...)")
        res_mask = rearrange(res_mask, "... -> (...)")

        res_labels = jax.nn.one_hot(labels, 23)
        res_cross_entropy = self._cross_entropy_loss(
            res_logits, res_labels  # , mask=res_mask
        )
        metrics["res_cross_entropy"] = res_cross_entropy.mean()
        total_loss += res_cross_entropy.mean()

        pred_labels = res_logits.argmax(-1)
        res_accuracy = pred_labels == labels
        res_accuracy = (res_accuracy * res_mask).sum() / (res_mask.sum() + 1e-6)
        res_accuracy = res_accuracy * (res_mask.sum() > 0).astype(res_accuracy.dtype)
        metrics["res_accuracy"] = res_accuracy

        # bound_labels = jax.nn.one_hot(ground.protein_data.boundary_token, 3)
        # sos_labels, eos_labels = bound_labels[..., -2], bound_labels[..., -1]
        # sos_cross_entropy = self._cross_entropy_loss(sos_logits, sos_labels)
        # eos_cross_entropy = self._cross_entropy_loss(eos_logits, eos_labels)
        # boundary_cross_entropy = sos_cross_entropy.mean() + eos_cross_entropy.mean()
        # metrics["boundary_cross_entropy"] = boundary_cross_entropy
        # total_loss += boundary_cross_entropy

        # pred_seq_len = eos_logits.argmax(-1) - sos_logits.argmax(-1)
        # metrics["avg_pred_seq_len"] = pred_seq_len.mean()
        # metrics["avg_gnd_seq_len"] = (labels > 0).sum(-1).mean()

        return model_output, total_loss, metrics
