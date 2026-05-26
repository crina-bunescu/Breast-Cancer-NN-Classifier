function s = sigmoid(y)
% Functia sigmoid - transforma orice numar intr-o probabilitate intre 0 si 1
% Intrare: y - numar, vector sau matrice
% Iesire:  s - aceleasi dimensiuni, valori intre 0 si 1
    s = 1 ./ (1 + exp(-y));
end