% Solver for the Time-Dependent Schrodinger Equation
% Homogeneous Dirichlet BCs
% Crank-Nicolson 2nd order method

% Domain Parameters
N = 500;
L = 10;
dx = L/N;

% Physical constants
hbar = 0.1;
m = 1;

% Time step
dt = 0.1;

% Complex constants from Schrodinger's Equation
k1 = 1i*hbar*dt / (4*m*dx*dx);
k2 = -1i*dt/(2*hbar);

% Initial condition (t=0)
f0 = @(x,y) exp(-((x-5).^2+(y-3).^2))*exp(1i*x+2i*y);

% Potential function
vf = @(x,y) 1e-1*max(0,sin(x+y));

fprintf('Generating mesh . . .\n');
tic;

[X,Y] = meshgrid(linspace(0,L,N),linspace(0,L,N));

toc;
fprintf('Generating IC . . .\n');
tic;

% Eval IC on mesh
Z = arrayfun(f0, X, Y);
Z = 1.0/sqrt(sum(sum(abs(Z).^2))*dx*dx)*Z;

% U is a column vector representing the current state at every mesh point.
U = sparse(reshape(Z,N*N,1));

toc;
fprintf('Generating V . . .\n');
tic;

% Eval V on mesh (and keep a 2D matrix around for plotting)
Vviz = arrayfun(vf, X, Y);
V = spdiags(reshape(Vviz,N*N,1), [0], N*N, N*N);

toc;
fprintf('Generating sparse simulation matrices . . .\n');
tic;

% A rather hack-ish way of generating the Laplacian matrix
% Look into delsq() in the future.
offdiag = repmat([repmat([1],1,N-1),0],1,N);
offdiag = offdiag(1:N*N);
offdiag2 = ones(1,N*N);
maindiag = -4*offdiag2;

D = spdiags([offdiag2' offdiag' maindiag' circshift(offdiag',1) offdiag2'], ...
    [-N -1 0 1 N], N*N, N*N);

% The matrix inv(B)*A advances the state of the simulation.
A = (speye(N*N)+k1.*D+k2.*V);
B = (speye(N*N)-k1.*D-k2.*V);

% This will store a log of our total probability amplitude as the
% simulation progresses. This is plotted at the end to verify conservation
% of probability.
maglist = [];

toc;

fprintf('Running Simulation . . .\n');

tic;
for i=1:5000
    % Use Conjugate Gradients Squared method to advance state.
    [U,flag] = cgs(B,(A*U));

    % Plot probability amplitude
    surf(X,Y,abs(reshape(full(U),N,N)).^2,'EdgeColor','none');
    hold on;
    % Uncomment these lines to plot the potential function
    %h=mesh(X,Y,5*Vviz,'FaceColor','black','EdgeColor','black');
    %alpha(h,0.1);
    axis([0 L 0 L 0 1]);
    xlabel('x');
    ylabel('y');
    grid on;
    hold off;
    drawnow;
    maglist = [maglist, sum(abs(full(U)).^2)*dx*dx];
    % Log iteration # periodically for reference.
    if (mod(i,100) == 0)
        fprintf('Iter #: %f\n',i);
    end
end
toc;
% Plot total prob. amplitude vs. iteration.
plot(maglist);
title('Total Probability Amplitude wrt. Iteration #');
xlabel('Iteration #');
ylabel('Total Probability');
grid on;
