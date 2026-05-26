%% Citirea si pregatirea datelor
clc; clear; close all;

% Citim fisierul wdbc.data si il punem intr-un tabel MATLAB
data = readtable('wdbc.data', 'FileType', 'text', 'Delimiter', ',', 'HeaderLines', 0);

% Extragem coloana 2 care contine etichetele M sau B si o punem intr-o matrice
etichete = table2array(data(:, 2));

% Extragem coloanele 3-32 care contin cele 30 de caracteristici numerice
A = table2array(data(:, 3:32));

% Codificam etichetele M/B in numere 1/0
e = zeros(size(etichete, 1), 1);

for i = 1:length(etichete)
    % daca eticheta i este 'M' (malign) punem 1
    if strcmp(etichete{i}, 'M')
        e(i) = 1;
    % daca eticheta i este 'B' (benign) punem 0
    else
        e(i) = 0;
    end
end

% Impartim datele in train (80%) si test (20%)
N = size(A, 1);

N_train = round(0.8 * N);
N_test = N - N_train;

A_train = A(1:N_train, :);
A_test = A((N_train + 1):end, :);

% Facem acelasi lucru pentru vectorul de etichete
e_train = e(1:N_train);
e_test = e((N_train + 1):end);

% Normalizam datele folosind formula min-max
A_min = min(A_train);
A_max = max(A_train);

% Impartim la (A_max - A_min) ca sa obtinem valori intre 0 si 1
A_train_norm = (A_train - A_min) ./ (A_max - A_min);
A_test_norm = (A_test - A_min) ./ (A_max - A_min);

% Adaugam coloana de bias (coloana de 1-uri)
A_train_bar = [A_train_norm, ones(N_train, 1)];
A_test_bar = [A_test_norm, ones(N_test, 1)];

%% Definirea retelei neuronale

% Stabilim dimensiunile retelei
n = size(A_train_bar, 2);   % numarul de coloane din A_train_bar = 31 (30 caracteristici + 1 bias)
m = 15;                      % numarul de neuroni pe stratul ascuns

% Initializam parametrii cu valori mici aleatoare
% fiecare coloana din X reprezinta ponderile unui neuron de pe stratul ascuns
% inmultim cu 0.01 ca sa pornim cu valori mici
X = 0.01 * randn(n, m);

% fiecare coloana din x reprezinta ponderile stratului de iesire
x = 0.01 * randn(m, 1);

%% Metoda Gradient cu Pas Variabil

% Parametrii algoritmului
alpha_init = 1;
factor_micsorare = 0.5;     % injumatatim la fiecare incercare
c1 = 0.0001;                % conditia Armijo - vrem orice scadere cat de mica
epsilon = 1e-4;
maxIter = 2000;

% Initializam parametrii (pornim din acelasi punct)
x_gd = x;
X_gd = X;

% Vectori pentru stocarea evolutiei (pentru grafice)
pierdere_gd = zeros(maxIter, 1);       % valorile functiei de pierdere
grad_norm_gd = zeros(maxIter, 1);      % norma gradientului
timp_gd = zeros(maxIter, 1);           % timpul la fiecare iteratie

fprintf('Incepe Metoda Gradient cu Backtracking...\n');
t_start_gd = tic;  % pornim cronometrul

for k = 1:maxIter

    % Calculam functia de pierdere si gradientii
    [L, grad_x, grad_X] = functie_pierdere(x_gd, X_gd, A_train_bar, e_train);

    % Calculam norma gradientului (concatenam toti parametrii intr-un singur vector ca sa calculam norma totala)
    grad_total = [grad_x; grad_X(:)];
    norma_grad = norm(grad_total);

    % Salvam valorile pentru grafice
    pierdere_gd(k) = L;
    grad_norm_gd(k) = norma_grad;
    timp_gd(k) = toc(t_start_gd);

    % criteriu de oprire
    if norma_grad < epsilon
        fprintf('Backtracking a convers la iteratia %d\n', k);
        pierdere_gd = pierdere_gd(1:k);
        grad_norm_gd = grad_norm_gd(1:k);
        timp_gd = timp_gd(1:k);
        break;
    end

    % Backtracking - cautam un pas bun
    alpha = alpha_init;

    % Calculam valoarea functiei dupa un pas potential
    x_nou = x_gd - alpha * grad_x;
    X_nou = X_gd - alpha * grad_X;
    L_nou = functie_pierdere(x_nou, X_nou, A_train_bar, e_train);

    % Cat timp conditia Armijo nu e satisfacuta -> micsoram pasul
    while L_nou > L - c1 * alpha * norma_grad^2
        alpha = factor_micsorare * alpha;   % injumatatim pasul
        x_nou = x_gd - alpha * grad_x;
        X_nou = X_gd - alpha * grad_X;
        L_nou = functie_pierdere(x_nou, X_nou, A_train_bar, e_train);
    end

    % Actualizam parametrii cu pasul acceptat
    x_gd = x_gd - alpha * grad_x;
    X_gd = X_gd - alpha * grad_X;

end

fprintf('Pierdere finala Backtracking: %.4f\n', pierdere_gd(end));
fprintf('Timp total Backtracking: %.2f secunde\n', timp_gd(end));


%% Metoda Gradient Stocastica (SGD)

% Parametrii algoritmului
alpha_sgd = 0.1;      % pas fix de invatare
epsilon_sgd = 1e-4;
maxIter_sgd = 2000;

% Initializam parametrii din acelasi punct de start
x_sgd = x;
X_sgd = X;

% Vectori pentru stocarea evolutiei
pierdere_sgd = zeros(maxIter_sgd, 1);
grad_norm_sgd = zeros(maxIter_sgd, 1);
timp_sgd = zeros(maxIter_sgd, 1);

fprintf('Incepe Metoda SGD...\n');
t_start_sgd = tic;

for k = 1:maxIter_sgd

    % Alegem un singur exemplu aleator din setul de antrenare
    i = randi(N_train);

    % Extragem doar exemplul i din datele de antrenare
    a_i = A_train_bar(i, :);
    e_i = e_train(i);

    % Calculam gradientul doar pe exemplul i
    [~, grad_x_i, grad_X_i] = functie_pierdere(x_sgd, X_sgd, a_i, e_i);

    % Actualizam parametrii folosind gradientul partial
    x_sgd = x_sgd - alpha_sgd * grad_x_i;
    X_sgd = X_sgd - alpha_sgd * grad_X_i;

    % Calculam pierderea si norma gradientului pe intreg setul
    [L_sgd, grad_x_full, grad_X_full] = functie_pierdere(x_sgd, X_sgd, A_train_bar, e_train);
    grad_total_sgd = [grad_x_full; grad_X_full(:)];
    norma_grad_sgd = norm(grad_total_sgd);

    % Salvam valorile pentru grafice
    pierdere_sgd(k) = L_sgd;
    grad_norm_sgd(k) = norma_grad_sgd;
    timp_sgd(k) = toc(t_start_sgd);

    % criteriu de oprire
    if norma_grad_sgd < epsilon_sgd
        fprintf('SGD a convers la iteratia %d\n', k);
        pierdere_sgd = pierdere_sgd(1:k);
        grad_norm_sgd = grad_norm_sgd(1:k);
        timp_sgd = timp_sgd(1:k);
        break;
    end

end

fprintf('Pierdere finala SGD: %.4f\n', pierdere_sgd(end));
fprintf('Timp total SGD: %.2f secunde\n', timp_sgd(end));

%% Grafice comparative

% Grafic 1: Functia de pierdere de-a lungul iteratiilor
figure;
plot(1:length(pierdere_gd), pierdere_gd, 'b-', 'LineWidth', 2);
hold on;
plot(1:length(pierdere_sgd), pierdere_sgd, 'r-', 'LineWidth', 2);
xlabel('Iteratia');
ylabel('Functia de pierdere');
title('Evolutia functiei de pierdere');
legend('Gradient Backtracking', 'SGD');
grid on;

% Grafic 2: Norma gradientului de-a lungul iteratiilor
figure;
semilogy(1:length(grad_norm_gd), grad_norm_gd, 'b-', 'LineWidth', 2);
hold on;
semilogy(1:length(grad_norm_sgd), grad_norm_sgd, 'r-', 'LineWidth', 2);
xlabel('Iteratia');
ylabel('Norma gradientului (scala logaritmica)');
title('Evolutia normei gradientului');
legend('Gradient Backtracking', 'SGD');
grid on;

% Grafic 3: Functia de pierdere in functie de timp
figure;
plot(timp_gd, pierdere_gd, 'b-', 'LineWidth', 2);
hold on;
plot(timp_sgd, pierdere_sgd, 'r-', 'LineWidth', 2);
xlabel('Timp (secunde)');
ylabel('Functia de pierdere');
title('Evolutia pierderii in functie de timp');
legend('Gradient Backtracking', 'SGD');
grid on;

%% Evaluarea performantei pe datele de test

% Facem predictii cu parametrii obtinuti de fiecare metoda

% A. Folosim parametrii de la Backtracking
Z_test_gd = A_test_bar * X_gd;
H_test_gd = softsign(Z_test_gd);
y_test_gd = H_test_gd * x_gd;
y_hat_test_gd = sigmoid(y_test_gd);

% Aplicam pragul de decizie 0.5
% daca probabilitatea e >= 0.5 -> clasa 1 (malign)
% daca probabilitatea e < 0.5  -> clasa 0 (benign)
predictii_gd = double(y_hat_test_gd >= 0.5);

% B. Folosim parametrii de la SGD
Z_test_sgd = A_test_bar * X_sgd;
H_test_sgd = softsign(Z_test_sgd);
y_test_sgd = H_test_sgd * x_sgd;
y_hat_test_sgd = sigmoid(y_test_sgd);
predictii_sgd = double(y_hat_test_sgd >= 0.5);

% Matricea de confuzie pentru fiecare metoda
CM_gd = confusionmat(e_test, predictii_gd);
CM_sgd = confusionmat(e_test, predictii_sgd);

% Calculam metricile de performanta pentru Backtracking
RP_gd = CM_gd(2,2);   % Real Pozitiv
FN_gd = CM_gd(2,1);   % Fals Negativ
FP_gd = CM_gd(1,2);   % Fals Pozitiv
RN_gd = CM_gd(1,1);   % Real Negativ

acuratete_gd = (RP_gd + RN_gd) / (RP_gd + RN_gd + FP_gd + FN_gd);
precizie_gd = RP_gd / (RP_gd + FP_gd);
sensibilitate_gd = RP_gd / (RP_gd + FN_gd);
specificitate_gd = RN_gd / (RN_gd + FP_gd);
f1_gd = 2 * precizie_gd * sensibilitate_gd / (precizie_gd + sensibilitate_gd);

% Calculam metricile de performanta pentru SGD
RP_sgd = CM_sgd(2,2);
FN_sgd = CM_sgd(2,1);
FP_sgd = CM_sgd(1,2);
RN_sgd = CM_sgd(1,1);

acuratete_sgd = (RP_sgd + RN_sgd) / (RP_sgd + RN_sgd + FP_sgd + FN_sgd);
precizie_sgd = RP_sgd / (RP_sgd + FP_sgd);
sensibilitate_sgd = RP_sgd / (RP_sgd + FN_sgd);
specificitate_sgd = RN_sgd / (RN_sgd + FP_sgd);
f1_sgd = 2 * precizie_sgd * sensibilitate_sgd / (precizie_sgd + sensibilitate_sgd);

%% Afisam rezultatele
fprintf('\n===== REZULTATE BACKTRACKING =====\n');
fprintf('Matricea de confuzie:\n');
disp(CM_gd);
fprintf('Acuratete:     %.4f\n', acuratete_gd);
fprintf('Precizie:      %.4f\n', precizie_gd);
fprintf('Sensibilitate: %.4f\n', sensibilitate_gd);
fprintf('Specificitate: %.4f\n', specificitate_gd);
fprintf('F1-score:      %.4f\n', f1_gd);

fprintf('\n===== REZULTATE SGD =====\n');
fprintf('Matricea de confuzie:\n');
disp(CM_sgd);
fprintf('Acuratete:     %.4f\n', acuratete_sgd);
fprintf('Precizie:      %.4f\n', precizie_sgd);
fprintf('Sensibilitate: %.4f\n', sensibilitate_sgd);
fprintf('Specificitate: %.4f\n', specificitate_sgd);
fprintf('F1-score:      %.4f\n', f1_sgd);
