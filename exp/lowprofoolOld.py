import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
# from art.attacks.evasion import LowProFool
# from torch.autograd.gradcheck import zero_gradients

class LowProFool:
    """
    Custom LowProFool attack implementation.

    This class implements the LowProFool adversarial attack with a feature 
    importance calculation method and integration with the repository's framework.
    """

    def __init__(self, classifier, max_iter=50, alpha=0.01, lambda_=0.1, bounds=(0, 1)):
        """
        Initialize the LowProFool attack.

        Parameters:
            classifier: The target model (ART-compatible classifier).
            max_iter (int): Maximum number of iterations.
            alpha (float): Step size for perturbation updates.
            lambda_ (float): Weight for the L2 norm in the loss function.
            bounds (tuple): Bounds for clipping adversarial examples (default: (0, 1)).
        """
        self.classifier = classifier
        self.max_iter = max_iter
        self.alpha = alpha
        self.lambda_ = lambda_
        self.bounds = bounds


    def generate(self, x, y=None, feature_importance_method='pearson'):
        """
        Generate adversarial examples using the LowProFool attack.

        Parameters:
            x (np.ndarray or torch.Tensor): Input samples.
            y (np.ndarray or torch.Tensor): One-hot encoded target labels (optional).
            feature_importance_method (str): Method to calculate feature importance ('shap' or 'pearson').

        Returns:
            np.ndarray: Adversarial examples for the input samples.
        """
        if y is None:
            raise ValueError("Target labels `y` must be provided for the generate method.")

        # if not isinstance(x, torch.Tensor):
        #     x = torch.FloatTensor(x)
        # if not isinstance(y, torch.Tensor):
        #     y = torch.FloatTensor(y)

        # Calculate feature importance
        # x_np, y_np = x.cpu().numpy(), y.cpu().numpy()  # Convert to numpy for feature importance calculation
        # weights = self.calculate_feature_importance(x_np, y_np, method=feature_importance_method)
        
        weights = self.calculate_feature_importance(x, y, method=feature_importance_method)
        # Initialize storage for adversarial examples
        adversarial_examples = []
        original_preds = []
        final_preds = []
        # x = torch.tensor(x, dtype=torch.float32)
        # y = torch.tensor(y, dtype=torch.int64)
        # Generate adversarial examples for each sample
        for i in range(x.shape[0]):

            x_sample = x[i]  # Ensure batch dimension for the sample
            # y_target = y[i]  # Target label for the current sample
            x_tensor = torch.FloatTensor(x_sample)

            # Generate adversarial example using LowProFool
            orig_pred, final_pred, adv_example, _ = self.lowProFool(
                x_tensor,
                model=self.classifier,
                weights=weights,
                bounds=self.bounds,
                max_iter=self.max_iter,
                alpha=self.alpha,
                lambda_=self.lambda_,
            )

            # Append results
            adversarial_examples.append(adv_example)
            original_preds.append(orig_pred)
            final_preds.append(final_pred)

        # Convert results to numpy arrays for consistency
        adversarial_examples = np.array(adversarial_examples)
        original_preds = np.array(original_preds)
        final_preds = np.array(final_preds)

        return adversarial_examples, original_preds, final_preds


    def lowProFool(self, x, model, weights, bounds, max_iter, alpha, lambda_):
        """
        Generates an adversarial examples x' from an original sample x

        :param x: tabular sample
        :param model: neural network
        :param weights: feature importance vector associated with the dataset at hand
        :param bounds: bounds of the datasets with respect to each feature
        :param max_iter: maximum number of iterations ran to generate the adversarial examples
        :param alpha: scaling factor used to control the growth of the perturbation
        :param lambda_: trade off factor between fooling the classifier and generating imperceptible adversarial example
        :return: original label prediction, final label prediction, adversarial examples x', iteration at which the class changed
        """
        # print(x)
        # print(x.shape)
        # x=x.unsqueeze(0)
        # print(x.shape)
        r = Variable(torch.FloatTensor(1e-4 * np.ones(x.numpy().shape)), requires_grad=True) 
        v = torch.FloatTensor(np.array(weights))
        
        output = model.predict((x + r).detach().cpu().numpy())
        output = torch.tensor(output, dtype=torch.float32)
        # if isinstance(output,np.ndarray):
        #     output = torch.tensor(output, dtype=torch.float32)
        
        # orig_pred = output.argmax(axis=1).item()
        orig_pred = output.max(0, keepdim=True)[1].cpu().numpy()
        target_pred = np.abs(1 - orig_pred)
        
        target = [0., 1.] if target_pred == 1 else [1., 0.]
        target = Variable(torch.tensor(target, requires_grad=False))
        print(f'{output.shape}--{orig_pred}--{target_pred}--{target.shape}')
        lambda_ = torch.tensor([lambda_])
        
        bce = nn.BCELoss()
        l1 = lambda v, r: torch.sum(torch.abs(v * r)) #L1 norm
        l2 = lambda v, r: torch.sqrt(torch.sum(torch.mul(v * r,v * r))) #L2 norm
        
        best_norm_weighted = np.inf
        best_pert_x = x
        
        loop_i, loop_change_class = 0, 0
        while loop_i < max_iter:
            
            if r.grad is not None:
                r.grad.zero_()

            # Computing loss 
            loss_1 = bce(output, target)
            loss_2 = l2(v, r)
            loss = (loss_1 + lambda_ * loss_2)

            # Get the gradient
            loss.backward(retain_graph=True)
            grad_r = r.grad.data.cpu().numpy().copy()
            
            # Guide perturbation to the negative of the gradient
            ri = - grad_r
        
            # limit huge step
            ri *= alpha

            # Adds new perturbation to total perturbation
            r = r.clone().detach().cpu().numpy() + ri
            
            # For later computation
            r_norm_weighted = np.sum(np.abs(r * weights))
            
            # Ready to feed the model
            r = Variable(torch.FloatTensor(r), requires_grad=True) 
            
            # Compute adversarial example
            xprime = x + r
            
            # Clip to stay in legitimate bounds
            xprime = LowProFool.clip(xprime, bounds[0], bounds[1])
            
            # Classify adversarial example
            # output = model.forward(xprime)
            # output_pred = output.max(0, keepdim=True)[1].cpu().numpy()
            output_pred = np.argmax(model.predict(xprime.detach().cpu().numpy()), axis=1)
            
            # Keep the best adverse at each iterations
            if output_pred != orig_pred and r_norm_weighted < best_norm_weighted:
                best_norm_weighted = r_norm_weighted
                best_pert_x = xprime

            if output_pred == orig_pred:
                loop_change_class += 1
                
            loop_i += 1 
            
        # Clip at the end no matter what
        best_pert_x = LowProFool.clip(best_pert_x, bounds[0], bounds[1])
        output = model.predict(best_pert_x.detach().cpu().numpy())
        output_pred = output.max(0, keepdim=True)[1].cpu().numpy()
        # output_pred = np.argmax(model.predict(best_pert_x.detach().cpu().numpy()), axis=1)

        return orig_pred, output_pred, best_pert_x.clone().detach().cpu().numpy(), loop_change_class 

    def calculate_feature_importance(self, x, y, method='pearson'):
        """
        Calculate feature importance using SHAP or Pearson correlation.

        Parameters:
            x (np.ndarray): Input samples.
            y (np.ndarray): Labels for the input samples.
            method (str): Method for feature importance ('shap' or 'pearson').

        Returns:
            np.ndarray: Feature importance scores.
        """
        if method == 'shap':
            import shap
            explainer = shap.Explainer(self.classifier.predict, x)
            shap_values = explainer.shap_values(x[:100])  # Use first 100 samples for efficiency
            importance = np.mean(np.abs(shap_values), axis=1)
        elif method == 'pearson':
            from scipy.stats import pearsonr
            importance = []
            for i in range(x.shape[1]):  # Iterate over features
                corr, _ = pearsonr(x[:, i], y)
                importance.append(abs(corr))
            importance = np.array(importance)
        else:
            raise ValueError("Invalid method. Choose 'shap' or 'pearson'.")

        return importance
    
    @staticmethod
    def clip(current, low_bound, up_bound):
        assert(len(current) == len(up_bound) and len(low_bound) == len(up_bound))
        low_bound = torch.FloatTensor(low_bound)
        up_bound = torch.FloatTensor(up_bound)
        clipped = torch.max(torch.min(current, up_bound), low_bound)
        return clipped
