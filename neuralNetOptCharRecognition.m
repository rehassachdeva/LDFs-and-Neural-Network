% Open the file
fileID = fopen('optdigits-orig.tra', 'r');

% Read the text
C = textscan(fileID, '%s');

% This gives C as a column vector of lines as strings
C = C{1};

numLines = size(C, 1);

% Initialize sample inputs and results as NULL matrices
sample = [];
res = [];

% Digits chosen 3, 5 and 7 denoted by output 00, 01 and 10 respectively
for i = 33:33:numLines
    
    if (C{i} == '3')
        sample = [sample; C((i-32):(i-1))];
        res = [res; [0 0]];
    end
    if(C{i} == '5')
        sample = [sample; C((i-32):(i-1))];
        res = [res; [0 1]];
    end
    if(C{i} == '7')
        sample = [sample; C((i-32):(i-1))];
        res = [res; [1 0]];
    end
    
end

% Initialize samples for storing inputs in double data type
samples = zeros(size(sample, 1), 32);

% This converts lines into row vectors
sample = cell2mat(sample);

for i = 1:size(sample, 1)
    
    for j = 1:32
        % Convert strings to double data type
        samples(i, j) = str2double(sample(i, j));
    end
    
end

% Initialize sample as NULL matrix, to be used for storing 8X8 bitmaps as
% row vectors
sample = [];

for i = 1:32:size(samples, 1) - 31
    
    % Downsize 32X32 to 8X8.
    % Bilinear uses weighted average of 2X2 windows to avoid aliasing
    temp = imresize(samples((i:i+31), :), 0.25, 'bilinear');
    
    % Append 8X8 bitmap as a row vector of size 64
    sample = [sample; temp(:)'];
    
end

% Apppend x0=1 to each sample to account for bias weights
sample = [sample ones(size(sample, 1), 1) ];

% Initialize error to some large value
error = 10000;

% numX stores number of inputs (65 in this case)
numX = size(sample, 2);

% numNH stores number of hidden units (generally numX/10)
numNH = 6;

% numZ stores number of outputs (2 in this case)
numZ = 2;

% eta is the convergence rate. Chosen optimally by hit and trial
eta = 0.1;

% dimensionality of input data is 64
dim = 64;

lim = 1 / sqrt(numX);

% w1 stores the weights from input to hidden layer. Chosen optimally between
% -1/root(dim) to +1/root(dim)
w1 = -lim + 2 * lim * rand(numX, numNH);

lim = 1 / sqrt(numNH);

% w2 stores the weights from hidden to output layer. Chosen optimally
% between -1/root(nH) to +1/root(nH)
w2 = -lim + 2 * lim * rand(numNH + 1, numZ);

% keep a limit on maximum number of iterations just in case we diverge
MAX_ITER = 10000;

iter = 0;

while(error > 0.1 && iter < MAX_ITER)
    
    % Increment iteration count
    iter = iter + 1;
    
    % Calculate net for each hidden unit for each input
    netJ = sample * w1;
    
    % Apply activation function sigmoid to netJ
    yJ = 1./(1 + exp(-netJ));
    
    % Apppend y0=1 to each output to account for bias weights for hidden to
    % output layer
    yJ = [yJ ones(size(yJ,1),1) ];
    
    % Calculate net for each hidden unit for each output
    netK = yJ*w2;
    
    % Apply activation function sigmoid to netK
    zK = 1./(1 + exp(-netK));
    
    % Compute error as 0.5*||t-z||^2
    error = 0.5.*((norm(res - zK))^2)
    
    % Compute delta2 as (t-z)*f'(netK) and we know for sigmoid f'=f(1-f)
    delta2 = (res - zK).*(zK.*(1 - zK));
    
    % Compute delta1 as summation over k(delta2*w2)*f'(netJ)
    delta1 = (delta2*w2').*(yJ.*(1 - yJ));
    
    % Discard the last column as it is extra, due to bias weights from
    % hidden to output layer
    delta1(:,size(delta1,2)) = [];
    
    % Change in weights is given by eta.delta.x and eta.delta.y    
    dw1 = eta.*(sample'*delta1);
    dw2 = eta.*(yJ'*delta2);
    
    % Compute new weights
    w1 = w1 + dw1;
    w2 = w2 + dw2;
    
end

w1;
w2;