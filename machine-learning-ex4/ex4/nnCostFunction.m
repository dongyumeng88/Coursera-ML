function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
% hidden_layer_size = 25; input_layer_size = 400

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m,1) X]; % 5000x401 

z2 = X * Theta1'; % 5000x25
hz2= sigmoid(z2); % 5000x25
a2 = hz2;% 5000x25
a2 = [ones(m,1) a2]; % 5000x26
z3 = a2 * Theta2'; % 5000x10
hz3 = sigmoid(z3); % 5000x10
a3 = hz3; % 5000x10

% YD: convert y from label vector to 0,1 matrix!!
yvec = zeros(num_labels,m);
for c = 1:num_labels
    pos = find(y==c);
    yvec(c,pos) = 1;
end

% YD: Use matrix will give you wrong J result
% J = 1/m * (-yvec * log(a3) - (1-yvec) * log(1-a3));

Cost = -yvec' .* log(a3) - (1-yvec') .* log(1-a3);
J = 1/m * sum(sum(Cost));


%-----------
% reqularized cost function
%-----------

Theta1_nobias = Theta1(:,2:end);
Theta2_nobias = Theta2(:,2:end);

Regu = lambda/(2*m) * (sum(sum(Theta1_nobias.^2)) + sum(sum(Theta2_nobias.^2)));

J = 1/m * sum(sum(Cost)) + Regu;

%-----------
% calculate gradients
%-----------
UC_delta1 = 0;
UC_delta2 = 0;
delta3 = a3 - yvec'; % 5000 x 10
z2 = [ones(m,1) z2]; % previous z2 is 5000x25. But Theta2 is 10x26. So need ones.
delta2 = delta3 * Theta2 .*sigmoidGradient(z2); % 5000x26
% no need to calcualte delta1 (input layer) 
delta2 = delta2(:,2:end); % 5000x25

UC_delta2 = UC_delta2 + delta3'*a2; % should be same dimension as Theta2, 10x26
UC_delta1 = UC_delta1 + delta2'*X; % should be same dimension as Theta1, 25x401

Theta1_grad = (1/m) * UC_delta1; % 25x401
Theta2_grad = (1/m) * UC_delta2; % 10x26
 
Theta1_reg = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_reg = [zeros(size(Theta2,1),1) Theta2(:,2:end)];

Theta1_grad = (1/m) * UC_delta1 + (lambda/m)*Theta1_reg; % 25x401
Theta2_grad = (1/m) * UC_delta2 + (lambda/m)*Theta2_reg; % 10x26

%for t = 1:m
%    delta2 = Theta2' * delta3(t,:)' .* sigmoidGradient()    
%end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
