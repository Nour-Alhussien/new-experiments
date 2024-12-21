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

        weights = self.calculate_feature_importance(x, y, method=feature_importance_method)

        # Initialize storage for adversarial examples
        adversarial_examples = []
        original_preds = []
        final_preds = []
        
        # Generate adversarial examples for each sample
        for i in range(x.shape[0]):

            x_sample = x[i]  # Ensure batch dimension for the sample
            y_target = y[i]  # Target label for the current sample


            # Generate adversarial example using LowProFool
            orig_pred, final_pred, adv_example, _ = self.lowProFool(
                x_sample,
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

        :param x: tabular sample (NumPy array)
        :param model: TensorFlowV2Classifier
        :param weights: feature importance vector associated with the dataset at hand
        :param bounds: bounds of the datasets with respect to each feature
        :param max_iter: maximum number of iterations ran to generate the adversarial examples
        :param alpha: scaling factor used to control the growth of the perturbation
        :param lambda_: trade off factor between fooling the classifier and generating imperceptible adversarial example
        :return: original label prediction, final label prediction, adversarial examples x', iteration at which the class changed
        """

        # Initial perturbation and feature weights
        r = np.zeros_like(x)
        v = np.array(weights)

        # Initial prediction
        x_with_r = np.expand_dims(x + r, axis=0)  # Add batch dimension
        output = model.predict(x_with_r)[0]  # Model prediction
        orig_pred = np.argmax(output)
        target_pred = 1 - orig_pred

        # Create one-hot target
        target = np.eye(len(output))[target_pred].astype(np.float32)  # One-hot target vector

        best_norm_weighted = np.inf
        best_pert_x = x.copy()
        loop_i, loop_change_class = 0, 0

        while loop_i < max_iter:
            # Compute loss gradient
            x_with_r = np.expand_dims(x + r, axis=0)
            gradient = model.loss_gradient(x_with_r, np.expand_dims(target, axis=0))[0]  # Gradient w.r.t. x

            # Update perturbation
            r -= alpha * gradient

            # Apply regularization
            r_norm_weighted = np.sum(np.abs(r * v))
            # r = LowProFool.clip(r, -np.inf, np.inf)  # Optional regularization bounds

            # Compute adversarial example
            xprime = x + r
            xprime = np.clip(xprime, bounds[0], bounds[1])  # Clip to feature bounds

            # Classify adversarial example
            xprime_with_batch = np.expand_dims(xprime, axis=0)
            output_pred = np.argmax(model.predict(xprime_with_batch)[0])

            # Keep the best adversarial example
            if output_pred != orig_pred and r_norm_weighted < best_norm_weighted:
                best_norm_weighted = r_norm_weighted
                best_pert_x = xprime

            if output_pred == orig_pred:
                loop_change_class += 1

            loop_i += 1

        # Final clip and return
        best_pert_x = np.clip(best_pert_x, bounds[0], bounds[1])
        return orig_pred, output_pred, best_pert_x, loop_change_class


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
    
    # @staticmethod
    # def clip(current, low_bound, up_bound):
    #     """
    #     Clip the values of the input array to specified bounds.

    #     Parameters:
    #         current (np.ndarray): Input array to clip.
    #         low_bound (float or np.ndarray): Lower bounds for each feature.
    #         up_bound (float or np.ndarray): Upper bounds for each feature.

    #     Returns:
    #         np.ndarray: Clipped array.
    #     """
    #     assert len(current) == len(up_bound) and len(low_bound) == len(up_bound)
    #     return np.clip(current, low_bound, up_bound)

