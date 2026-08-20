from bhaskera.evaluation.registry import register_metric

@register_metric("token_accuracy")
class TokenAccuracyMetric:
    def compute(self, predictions, labels, losses) -> dict:
        if not predictions or not labels:
            return {"validation/token_accuracy": 0.0}
            
        correct, total = 0, 0
        for pred, label in zip(predictions, labels):
            mask = label != -100
            correct += (pred[mask] == label[mask]).sum().item()
            total += mask.sum().item()
            
        acc = correct / total if total > 0 else 0.0
        return {"validation/token_accuracy": acc}
