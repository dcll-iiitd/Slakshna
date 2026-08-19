import math
from bhaskera.evaluation.registry import register_metric

@register_metric("perplexity")
class PerplexityMetric:
    def compute(self, predictions, labels, losses) -> dict:
        if not losses:
            return {"validation/perplexity": float("inf")}
        mean_loss = sum(losses) / len(losses)
        try:
            ppl = math.exp(mean_loss)
        except OverflowError:
            ppl = float("inf")
        return {"validation/perplexity": ppl}
