int SolveFem::Tet4Isoparametric(MatrixXd eleCoord, MatrixXd& ke, MatrixXd zero)
{
    //Calculate the stiffness matrix of the eight-node isoparametric element with dianonal numbering element, not multiplied by E
    /*  Shape function for counterclockwise numbering element
        % N1 = 1 - s - t - n; % (0, 0, 0)
        % N2 = s; % (1, 0, 0)
        % N3 = t; % (0, 1, 0)
        % N4 = n; % (0, 0, 1)
    */

    ke.fill(0);

    //Elasticity Matrix D
    double lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = 1.0 / (2.0 * (1.0 + nu));
    MatrixXd D(6, 6);

    D << lambda + 2.0 * mu, lambda, lambda, 0.0, 0.0, 0.0, lambda, lambda + 2.0 * mu, lambda, 0.0, 0.0, 0.0,
        lambda, lambda, lambda + 2.0 * mu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, mu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu;

    MatrixXd  Ndiff(3, COL_2), Jacob(3, 3), dN(3, COL_2), B(6, 3 * COL_2);
    MatrixXd  J(3, 3);
    double detJ;

    //Calculte element stiffness matrix
    Ndiff << -1.0, 1.0, 0.0, 0.0,
        -1.0, 0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0, 1.0;
    //Jacob 3 * 3
    Jacob = Ndiff * eleCoord;
    J = Ndiff * eleCoord * eleCoord.transpose() * Ndiff.transpose();
    detJ = sqrt(J.determinant());
    //dN 3 * 4
    dN = Jacob.inverse() * Ndiff;
    //B 6 * 12
    B << dN.row(0), zero, zero, zero, dN.row(1), zero, zero, zero, dN.row(2), dN.row(1), dN.row(0), zero, zero, dN.row(2), dN.row(1), dN.row(2), zero, dN.row(0);
    //ke 12 * 12
    ke = ke + 1.0/6.0 * B.transpose() * D * B * detJ;

    return 0;
}