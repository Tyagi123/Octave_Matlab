function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regre/Users/gauravtyagi/Downloads/Workspace/Octave_Matlab/Octave_Matlab/machine-learning-ex2/ex2/costFunction.mssion
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
y1=sigmoid(X*theta);
J=sum((y- y1).^2)/(m);
for i = 1:size(theta)

  theta(i)= theta(i)-( sum(( y1 - y ).*X( :,i)))/m
  y2=sigmoid(X*theta);
  grad(i)=sum((y- y2).^2)/(m);

end
% =============================================================

end
