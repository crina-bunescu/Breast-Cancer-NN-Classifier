function [L, grad_x, grad_X] = functie_pierdere(x, X, A_bar, e)
% Calculeaza functia de pierdere si gradientii
% Intrari:
%   x     - ponderile stratului de iesire (m x 1)
%   X     - ponderile stratului ascuns (n+1 x m)
%   A_bar - datele de intrare cu bias (N x n+1)
%   e     - etichetele reale (N x 1)
% Iesiri:
%   L      - valoarea functiei de pierdere (un singur numar)
%   grad_x - gradientul fata de x (m x 1)
%   grad_X - gradientul fata de X (n+1 x m)

    % numarul de exemple
    N = size(A_bar, 1);

    % Calculam combinatia liniara a stratului ascuns
    Z = A_bar * X;

    % Aplicam SoftSign pentru a obtine iesirea stratului ascuns
    H = softsign(Z);

    % Calculam iesirea bruta a retelei
    y = H * x;

    % Aplicam sigmoid ca sa obtinem probabilitati intre 0 si 1
    y_hat = sigmoid(y);

    % Calculam functia de pierdere (entropia incrucisata binara)
    % + adaugam un epsilon mic ca sa evitam log(0) care e infinit
    epsilon = 1e-10;
    L = -1/N * sum(e .* log(y_hat + epsilon) + (1 - e) .* log(1 - y_hat + epsilon));

    % Calculam eroarea la iesire (diferenta dintre predictie si eticheta reala)
    delta = y_hat - e;

    % gradientul fata de x (ponderile stratului de iesire)
    grad_x = 1/N * (H' * delta);

    % gradientul fata de X (ponderile stratului ascuns)
    % mai intai calculam derivata SoftSign in punctele Z, apoi propagam eroarea inapoi prin stratul de iesire
    dZ = softsign_derivata(Z);
    delta_ascuns = (delta * x') .* dZ;
    grad_X = 1/N * (A_bar' * delta_ascuns);
end