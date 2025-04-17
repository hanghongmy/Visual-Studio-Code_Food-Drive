"""
    Monitoring utilities for training processes using Prometheus.
    It uses the 'Prometheus_client' library to expose metrics by visualizing in monitoring tools
    like Grafana.
    The metrics include:
    - Metrics Collection: tracks training progress, such as number of epochs and batches completed.
    - Prometheus Integration: starts an http server to expose metrics for Prometheus scraping.
    - Loss Tracking: records training and validation loss.
"""


from prometheus_client import Gauge, Counter, start_http_server, CollectorRegistry
import psutil
import os
import time


registry = CollectorRegistry()

class SystemMonitor:
    def __init__(self):
        self.memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application', registry=registry)
        self.cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage of the application', registry=registry)
        self.disk_usage = Gauge('app_disk_usage_bytes', 'Disk usage of the application', registry=registry)
    def update_metrics(self):
        """Update system resource metrics."""
        process = psutil.Process(os.getpid())
        self.memory_usage.set(process.memory_info().rss)
        self.cpu_usage.set(process.cpu_percent())
        self.disk_usage.set(psutil.disk_usage('/').used)

class TrainingMonitor:
    """Base class for tracking ML training metrics."""
    def __init__(self, port=8002, registry=registry):
        start_http_server(port, registry=registry)
        self.epoch = Counter('training_epoch_total', 'Total number of epochs completed', registry=registry)
        self.batch = Counter('training_batch_total', 'Total number of batches processed', registry=registry)
        self.loss = Gauge('training_loss', 'Current training loss', registry=registry)
        self.validation_loss = Gauge('validation_loss', 'Current validation loss', registry=registry)
        self.accuracy = Gauge('validation_accuracy', 'Current validation accuracy', registry=registry)
        self.custom_metric = Gauge('custom_metric_name', 'Description of the custom metric', registry=registry)
        self.flask_requests = Counter('flask_http_requests_total', 'Total HTTP requests handled by Flask', registry=registry)

    def record_epoch(self):
        """Increment epoch counter."""
        self.epoch.inc()

    def record_batch(self):
        """Increment batch counter."""
        self.batch.inc()

    def record_loss(self, loss):
        """Set the current training loss."""
        self.loss.set(loss)

    def record_validation_metrics(self, loss=None, accuracy=None):
        """Set validation loss and accuracy."""
        if loss is not None:
            self.validation_loss.set(loss)
        if accuracy is not None:
            self.accuracy.set(accuracy)
            
    def record_custom_metric(self, value):
        """Set a custom metric."""
        self.custom_metric.set(value)


class RegressionMonitor(TrainingMonitor):
    """Monitoring for regression models."""
    def __init__(self, port=8004):
        super().__init__(port)
        self.mse = Gauge('regression_mean_squared_error', 'Mean Squared Error')
        self.rmse = Gauge('regression_root_mean_squared_error', 'Root Mean Squared Error')
        self.mae = Gauge('regression_mean_absolute_error', 'Mean Absolute Error')
        self.r_squared = Gauge('regression_r_squared', 'R-squared coefficient')
        self.feature_importance = Gauge('feature_importance', 'Feature importance value', ['feature_name'])

        # Initialize a dictionary to store metrics
        self.metrics = {
            "mse": None,
            "rmse": None,
            "mae": None,
            "r_squared": None,
            "feature_importance": {}
        }

    def record_metrics(self, mse=None, rmse=None, mae=None, r_squared=None, feature_importance=None):
        """Record regression metrics."""
        if mse is not None:
            self.mse.set(mse)
            self.metrics["mse"] = mse
        if rmse is not None:
            self.rmse.set(rmse)
            self.metrics["rmse"] = rmse
        if mae is not None:
            self.mae.set(mae)
            self.metrics["mae"] = mae
        if r_squared is not None:
            self.r_squared.set(r_squared)
            self.metrics["r_squared"] = r_squared
        if feature_importance is not None:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            self.metrics["feature_importance"] = {feature_name: importance for feature_name, importance in sorted_features}
            for feature_name, importance in sorted_features:
                self.feature_importance.labels(feature_name=feature_name).set(importance)


    def reset_metrics(self):
        """Reset all regression metrics to their default state."""
        self.mse.set(0)
        self.rmse.set(0)
        self.mae.set(0)
        self.r_squared.set(0)
        self.metrics = {
            "mse": None,
            "rmse": None,
            "mae": None,
            "r_squared": None,
            "feature_importance": {}
        }

    def get_metrics(self):
        """Retrieve the current metrics."""
        return self.metrics


#class TreeModelMonitor(TrainingMonitor):
#    """Monitoring for tree-based models."""
#    def __init__(self, port=8005):
#        super().__init__(port)
#        self.tree_depth = Gauge('tree_max_depth', 'Maximum tree depth')
#        self.tree_leaves = Gauge('tree_leaf_count', 'Number of leaf nodes')
#        self.trees_count = Gauge('ensemble_tree_count', 'Number of trees in the ensemble')
#        self.boost_round = Counter('boosting_rounds_total', 'Total boosting rounds completed')
#        self.iteration_improvement = Gauge('iteration_improvement', 'Performance improvement in the last iteration')

#    def record_tree_metrics(self, depth=None, leaves=None, trees=None):
#        """Record tree structure metrics."""
#        if depth is not None:
#            self.tree_depth.set(depth)
#        if leaves is not None:
#            self.tree_leaves.set(leaves)
#        if trees is not None:
#            self.trees_count.set(trees)

#    def record_boost_round(self, improvement=None):
#        """Record a completed boosting round."""
#        self.boost_round.inc()
#        if improvement is not None:
#            self.iteration_improvement.set(improvement)

#    def reset_tree_metrics(self):
#        """Reset all tree-based metrics to their default state."""
#        self.tree_depth.set(0)
#        self.tree_leaves.set(0)
#        self.trees_count.set(0)
#        self.iteration_improvement.set(0)

if __name__ == "__main__":
    monitor = RegressionMonitor(port=8004)
    monitor.record_metrics(mse=0.1, rmse=0.316, mae=0.2, r_squared=0.95,
                            feature_importance={"feature1": 0.8, "feature2": 0.6})

    system_monitor = SystemMonitor()

    # Update system metrics periodically
    while True:
        system_monitor.update_metrics()
        time.sleep(5)