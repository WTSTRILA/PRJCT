from scipy.stats import ks_2samp
from functools import wraps


def detect_drift_decorator(train_samples_key='train_samples', test_samples_key='test_samples'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            model_evaluation_result = func(*args, **kwargs)

            train_samples = kwargs.get(train_samples_key)
            test_samples = kwargs.get(test_samples_key)

            if train_samples is not None and test_samples is not None:
                for i in range(train_samples.shape[1]):
                    train_feature = train_samples[:, i]
                    test_feature = test_samples[:, i]

                    statistic, p_value = ks_2samp(train_feature, test_feature)

                    alpha = 0.05
                    if p_value < alpha:
                        print(f"Drift detected in the feature {i} (p-value: {p_value:.4f})")
                    else:
                        print(f"Drift not detected in the feature {i} (p-value: {p_value:.4f})")

            return model_evaluation_result

        return wrapper

    return decorator
