% Initialize data samples and normalise them
w1 = [1 2 7;
    1 8 1;
    1 7 5;
    1 6 3;
    1 7 8;
    1 5 9;
    1 4 5
    ];

w2 = [1 4 2;
    1 -1 -1;
    1 1 3;
    1 3 -2;
    1 5 3.25;
    1 2 4;
    1 7 1
    ];

% Negate samples of one class
samples = [w1; -w2];

n = size(samples, 1);

MAX_ITER = 100000;

misclassified = true;

iter = 0;

% Keep some margin
margin = 1;

% Initialize weight vector
a = [1; 1; 1];

% Loop until there is a misclassified sample. Also keep a limit on maximum
% number of iterations in case we diverge
while(misclassified && iter < MAX_ITER)
    
    iter = iter + 1;
    misclassified = false;
    sampl = 1;
    
    % Find a sample which is misclassified and add it to weight vector
    while(sampl <= n)
    
        if(samples(sampl, :)*a < margin)
            
            a = a + samples(sampl, :)';
            misclassified=true;
            break;
            
        end
        
        sampl = sampl + 1;
    
    end
    
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