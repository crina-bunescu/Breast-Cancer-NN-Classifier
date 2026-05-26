function z = softsign(z_bar)
% Functia de activare SoftSign
% Intrare: z_bar - poate fi un numar, vector sau matrice
% Iesire:  z     - acelasi format ca intrarea, valori intre -1 si 1
    z = z_bar ./ (1 + abs(z_bar));
end