from bhaskera.evaluation.registry import register_metric

@register_metric("loss")
class LossMetric:
    def compute(self, predictions, labels, losses) -> dict:
        if not losses:
            return {"validation/loss": float("inf")}
        mean_loss = sum(losses) / len(losses)
        return {"validation/loss": mean_loss}
