function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hyp = X * theta;
diff = hyp - y;
diff_sqr = diff .* diff;
cost_unreg = (1/(2*m)) * sum(diff_sqr);
theta_no_bias = theta([2:size(theta, 1)],:);
theta_no_bias_sqr = theta_no_bias .* theta_no_bias;
reg_term = (lambda/(2*m)) * sum(theta_no_bias_sqr);
J = cost_unreg + reg_term;

grad_unreg = zeros(size(theta));
for j=1:n
  grad_unreg(j) = sum(diff .* X(:,j));
end
grad_reg = [0; theta([2:n],:)];
grad = ((1/m) * grad_unreg) + (lambda/m)*(grad_reg);
% =========================================================================

grad = grad(:);

end
