function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

c_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
c_len = length(c_vec);
sigma_len = length(sigma_vec);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
min_error = 100000;
for i=1:c_len
  c_i = c_vec(i);
  for j=1:sigma_len
    sig_j = sigma_vec(j);
    model_i_j = svmTrain(X, y, c_i, @(x1, x2) gaussianKernel(x1, x2, sig_j));
    pred_i_j = svmPredict(model_i_j, Xval);
    error_i_j = mean(double(pred_i_j ~= yval));
    if(min_error >= error_i_j)
      min_error = error_i_j;
      C = c_i;
      sigma = sig_j;
    end
  end
end



% =========================================================================

end
