import numpy as np
from mipha.framework import Aggregator


class HorizontalStackAggregator(Aggregator):
    """
    Simple aggregator stacking features horizontally.
    If all inputs are 3D, they are concatenated along the feature axis.
    Otherwise, all inputs are flattened to 2D.
    It is assumed that the first dimension is n_samples.
    """

    def __init__(self, component_name: str = None, always_flatten: bool = False):
        """
        :param component_name: Name of the component. Optional.
        :param always_flatten: If set to true, always flatten aggregated data to 2D.
        """
        super(HorizontalStackAggregator, self).__init__(component_name)
        self.always_flatten = always_flatten

    def aggregate_features(self, features):
        """
        :param features: List of features to be combined.
        :return: Aggregated features.
        """
        # Ensure that all inputs have the same number of samples
        num_samples = features[0].shape[0]
        if not all(f.shape[0] == num_samples for f in features):
            raise ValueError("All feature arrays must have the same number of samples")

        # Check if all inputs have the same dimension
        all_same_dimension = all(f.ndim == features[0].ndim for f in features)

        if not all_same_dimension or self.always_flatten:
            # Flatten each feature array to 2D if it has more than 2 dimensions and stack along the feature axis
            aggregated_features = self.flatten_features(features)
        else:
            # Concatenate all inputs along the last axis (feature axis)
            aggregated_features = np.concatenate(features, axis=-1)

        return aggregated_features

    @staticmethod
    def flatten_features(features):
        flattened_features = [f.reshape(f.shape[0], -1) if f.ndim > 2 else f for f in features]
        aggregated_features = np.hstack(flattened_features)
        return aggregated_features
