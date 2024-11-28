from collections import defaultdict
from itertools import combinations

import numpy as np
from zen_mapper.komplex import Simplex


class SimplicialComplex:
    def __init__(self):
        self._simplices = defaultdict(set)
        self.vertex_coords = {}
        self.vertex_functions = {}
        self.simplex_functions = {}

    def add_vertex(self, vertex_id, coords):
        """Add a vertex with coordinates"""
        self._simplices[0].add(Simplex([vertex_id]))
        self.vertex_coords[vertex_id] = np.array(coords)

    def add_simplex(self, vertices, dim):
        """Add a simplex and all its faces up to given dimension"""
        # Ensure vertices exist
        for vertex in vertices:
            if Simplex([vertex]) not in self._simplices[0]:
                raise ValueError(f"Vertex {vertex} not found in complex")

        # Create simplex using zen-mapper's Simplex class
        simplex = Simplex(vertices)

        # Add all faces up to dim
        for k in range(1, dim + 1):
            for face in combinations(vertices, k):
                self._simplices[k - 1].add(Simplex(face))

        # Add the simplex itself if its dimension matches dim
        if simplex.dim == dim:
            self._simplices[dim].add(simplex)

    def from_mapper_result(self, mapper_result, centroids):
        """Initialize from zen-mapper MapperResult"""
        # Add vertices with their centroids
        for i in range(len(mapper_result.nodes)):
            self.add_vertex(i, centroids[i])

        # Add simplices from nerve
        for dim in range(mapper_result.nerve.dim + 1):
            for simplex in mapper_result.nerve[dim]:
                self.add_simplex(simplex, dim + 1)

        return self

    def set_vertex_function(self, vertex_id, value):
        """Set function value at a vertex"""
        if Simplex([vertex_id]) not in self._simplices[0]:
            raise ValueError(
                f"Cannot assign vertex function value: vertex {vertex_id} "
                f"not found in complex"
            )
        self.vertex_functions[vertex_id] = value

    def extend_function(self, method="max"):
        """Extend vertex function to higher dimensional simplices"""
        for dim in range(1, max(self._simplices.keys()) + 1):
            for simplex in self._simplices[dim]:
                if method == "max":
                    self.simplex_functions[simplex] = max(
                        self.vertex_functions[v] for v in simplex
                    )
                elif method == "min":
                    self.simplex_functions[simplex] = min(
                        self.vertex_functions[v] for v in simplex
                    )
                elif method == "mean":
                    self.simplex_functions[simplex] = np.mean(
                        [self.vertex_functions[v] for v in simplex]
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

    @property
    def dimension(self):
        """Return the dimension of the complex"""
        return max(self._simplices.keys()) if self._simplices else -1

    def euler_characteristic(self, threshold=None):
        """
        Calculate the Euler characteristic considering only simplices with
        function values <= threshold

        Parameters:
        -----------
        threshold : float, optional
            If provided, only count simplices with function values <= threshold.
            If None, count all simplices.

        Returns:
        --------
        int
            The Euler characteristic
        """
        if threshold is None:
            return sum(
                (-1) ** k * len(simplices) for k, simplices in self._simplices.items()
            )

        chi = 0
        # For 0-simplices (vertices), use vertex_functions
        if 0 in self._simplices:
            vertex_count = sum(
                1
                for v in self._simplices[0]
                if self.vertex_functions.get(v[0], float("inf")) <= threshold
            )
            chi += vertex_count

        # For higher dimensional simplices, use simplex_functions
        for k in range(1, max(self._simplices.keys()) + 1):
            if k in self._simplices:
                simplex_count = sum(
                    1
                    for s in self._simplices[k]
                    if self.simplex_functions.get(s, float("inf")) <= threshold
                )
                chi += (-1) ** k * simplex_count

        return chi

    def validate(self):
        """Check if complex satisfies simplicial complex properties"""
        # Check that all faces of simplices are present
        for dim in range(1, self.dimension + 1):
            for simplex in self._simplices[dim]:
                # Check all faces
                for k in range(1, dim + 1):
                    for face in combinations(simplex, k):
                        face_simplex = Simplex(face)
                        if face_simplex not in self._simplices[k - 1]:
                            return False
        return True
