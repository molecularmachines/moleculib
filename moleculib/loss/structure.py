class VectorMapLoss(LossFunction):
    def __init__(
        self,
        weight=1.0,
        start_step=0,
        max_radius: float = 32.0,
        max_error: float = 800.0,
        norm_only=False,
    ):
        super().__init__(weight=weight, start_step=start_step)
        self.norm_only = norm_only
        self.max_radius = max_radius
        self.max_error = max_error

    def _call(
        self, rng_key, prediction: ProteinDatum, ground: ProteinDatum
    ) -> Tuple[ModelOutput, jax.Array, Dict[str, float]]:
        ground = ground[0]

        all_atom_coords = rearrange(
            prediction.datum["atom_coord"], "... a c -> (... a) c"
        )
        all_atom_coords_ground = rearrange(ground.atom_coord, "... a c -> (... a) c")
        all_atom_mask = rearrange(ground.atom_mask, "... a -> (... a)")

        vector_map = lambda x: rearrange(x, "i c -> i () c") - rearrange(
            x, "j c -> () j c"
        )

        cross_mask = rearrange(all_atom_mask, "i -> i ()") & rearrange(
            all_atom_mask, "j -> () j"
        )

        vector_maps = vector_map(all_atom_coords)
        vector_maps_ground = vector_map(all_atom_coords_ground)
        cross_mask = cross_mask & (safe_norm(vector_maps_ground) < self.max_radius)

        if self.norm_only:
            vector_maps = safe_norm(vector_maps)[..., None]
            vector_maps_ground = safe_norm(vector_maps_ground)[..., None]

        error = optax.huber_loss(vector_maps, vector_maps_ground, delta=1.0).mean(-1)
        if self.max_error > 0.0:
            error = jnp.clip(error, 0.0, self.max_error)

        error = (error * cross_mask.astype(error.dtype)).sum((-1, -2)) / (
            cross_mask.sum((-1, -2)) + 1e-6
        )
        error = error.mean()
        error = error * (cross_mask.sum() > 0).astype(error.dtype)

        metrics = dict(
            cross_vector_loss=error,
        )

        return prediction, error, metrics
