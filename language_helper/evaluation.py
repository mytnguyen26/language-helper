from torch import exp
from torch.nn import CrossEntropyLoss

class Evaluation:
    def __init__(self, loss):
        pass

    def perplexity(self):
        """
        
        Note that if we are using pytorch CrossEntropyLoss, the perplexity is
        simply calculated by applying torch.exp()
        credit: https://stackoverflow.com/questions/59209086/calculate-perplexity-in-pytorch
        """
        if isinstance(self.loss, CrossEntropyLoss):
            perplexity_score = exp(self.loss)
        return perplexity_score
    
    def blue():
        """
        BLUE is context-free evaluation metric for word based N-Grams.
        This method evaluate the quality of the prediction.
        """
        pass

    def run():
        pass