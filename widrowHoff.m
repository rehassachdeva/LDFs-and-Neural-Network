% Initialize data samples and normalise them
w1 = [1 2 7;
    1 8 1;
    1 7 5;
    1 6 3;
    1 7 8;
    1 5 9;
    1 4 5;
% Add these to make data linearly non-separable
%   1 2 4;
%   1 1 2
    ];

w2 = [1 4 2;
    1 -1 -1;
    1 1 3;
    1 3 -2;
    1 5 3.25;
    1 2 4;
    1 7 1;
% Add these to make data linearly non-separable
%   1 8 8;
%   1 7 8 
    ];

% Negate samples of one class
samples = [w1; -w2];

n = size(samples, 1);

misclassified = true;

% Initialize weight vector
a = [1; 1; 1];

% Initialize weight vector
margin = 0.1;

% Initialize convergence rate
convergenceRate = 1;

% Factor by which convergence rate decreases each iteration
annealingFactor = 0.9;

% Minimum allowed absolute change in weight vector
threshold = 0.0001;

change = norm(a);

% Loop until there is a misclassified sample and change is greater than
% threshold
while(misclassified && change > threshold)
    
    misclassified = false;
    sampl = 1;
    
    while(sampl <= n)
    
        if(samples(sampl, :)*a < margin)
            
            % gradient is given by y(b-ay)
            delta = samples(sampl, :)*(margin - samples(sampl, :)*a);
            
            % change in weight is eta times gradient
            change = norm(convergenceRate*delta');
            a = a + convergenceRate*delta';
            misclassified=true;
            
        end
        
        sampl = sampl + 1;
    
    end
    
    convergenceRate = convergenceRate*annealingFactor;
    
end

figure,

% Plot the data points
plot(w1(:,2), w1(:,3), 'r*');
hold on;
plot(w2(:,2), w2(:,3), 'bO');
hold on;

x = linspace(0,10,11);

% Equation of line is given by a*[1 x y]=0
y = (-a(3)/a(2))*x - a(1)/a(2);

plot(x,y,'g');