function dz = softsign_derivata(z_bar)
% Derivata functiei de activare SoftSign
% Intrare: z_bar - poate fi un numar, vector sau matrice
% Iesire:  dz    - acelasi format ca intrarea, valori intre 0 si 1
    dz = 1 ./ (1 + abs(z_bar)).^2;
end