class EarlyStoppingMonitor:
    def __init__(
        self,
        val_loss_patience=5,
        metric_patience=20,
        val_loss_min_delta=1e-4,
        metric_min_delta=1e-4,
        warmup_validations=8,
        history=None
    ):
        self.val_loss_patience = val_loss_patience
        self.metric_patience = metric_patience
        self.val_loss_min_delta = val_loss_min_delta
        self.metric_min_delta = metric_min_delta
        self.warmup_validations = warmup_validations
        self.best_val_loss = None
        self.best_metric = None
        self.val_loss_wait = 0
        self.metric_wait = 0
        self.validation_count = 0

        for epoch_summary in history or []:
            if epoch_summary.get("validated"):
                self.update(epoch_summary["epoch"], epoch_summary)

    def update(self, epoch, val_summary):
        if not val_summary.get("validated"):
            return self.create_status(epoch, False, None)

        self.validation_count += 1
        val_loss = val_summary["valLoss"]
        metric = val_summary["ssim"]
        val_loss_improved = self.best_val_loss is None or val_loss < self.best_val_loss - self.val_loss_min_delta
        metric_improved = self.best_metric is None or metric > self.best_metric + self.metric_min_delta

        if val_loss_improved:
            self.best_val_loss = val_loss
            self.val_loss_wait = 0
        else:
            self.val_loss_wait += 1

        if metric_improved:
            self.best_metric = metric
            self.metric_wait = 0
        else:
            self.metric_wait += 1

        reason = self.get_stop_reason()

        return self.create_status(epoch, reason is not None, reason)

    def get_stop_reason(self):
        if self.validation_count < self.warmup_validations:
            return None

        if self.val_loss_wait >= self.val_loss_patience:
            return f"valLoss did not improve for {self.val_loss_wait} validations"

        if self.metric_wait >= self.metric_patience:
            return f"best SSIM did not improve for {self.metric_wait} validations"

        return None

    def create_status(self, epoch, should_stop, reason):
        return {
            "epoch": epoch,
            "shouldStop": should_stop,
            "reason": reason,
            "bestValLoss": self.best_val_loss,
            "bestMetric": self.best_metric,
            "valLossWait": self.val_loss_wait,
            "metricWait": self.metric_wait,
            "validationCount": self.validation_count
        }


def create_early_stopping_monitor(args, history):
    return EarlyStoppingMonitor(
        val_loss_patience=args.val_loss_patience,
        metric_patience=args.metric_patience,
        val_loss_min_delta=args.val_loss_min_delta,
        metric_min_delta=args.metric_min_delta,
        warmup_validations=args.early_stop_warmup,
        history=history
    )
