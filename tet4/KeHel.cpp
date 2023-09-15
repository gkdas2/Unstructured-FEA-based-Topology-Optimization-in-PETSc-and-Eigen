int PDEFilter::KeHel(MatrixXd eleCoord, MatrixXd& KE, MatrixXd& T, PetscScalar rmin)
{
    //Helmholtz filter radius
    r = 1 / (2 * sqrt(3)) * rmin;

    KE.fill(0);
    T.fill(0);

    MatrixXd N(1, 4), dN(3, 4), J(3, 3), Dn(3, 4);
    MatrixXd Kd(3,3);
    Kd.fill(0);
    //isotropic filter
    Kd(0,0) = r * r,Kd(1,1) = r * r,Kd(2,2) = r * r;

    //Hammer integration points
    double a = (5.0 + 3.0 * sqrt(5.0)) / 20.0;
    double b = (5.0 - sqrt(5.0)) / 20.0;
    MatrixXd G(4, 3);
    G(0, 0) = a, G(0, 1) = b, G(0, 2) = b;
    G(1, 0) = b, G(1, 1) = a, G(1, 2) = b;
    G(2, 0) = b, G(2, 1) = b, G(2, 2) = a;
    G(3, 0) = b, G(3, 1) = b, G(3, 2) = b;

    for (int i = 0; i < 4; i++)
    {
        double s = G(i, 0);
        double t = G(i, 1);
        double v = G(i, 2);

        N << 1 - s - t - v,
            s,
            t,
            v;

        dN << -1.0, 1.0, 0.0, 0.0,
            -1.0, 0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0, 1.0;

        J = dN * eleCoord;

        Dn = J.inverse() * dN;

        KE = KE + 1.0 / 24.0 * (Dn.transpose() * Kd * Dn + N.transpose() * N) * J.determinant();
        T = T + 1.0 / 24.0 * N.transpose() * J.determinant();
    }

    return 0;
}