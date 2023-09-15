PetscErrorCode SolveFem::ComputeStress(Vec XPhys, PetscScalar Emax, MatrixXd eleXCoord, MatrixXd eleYCoord, MatrixXd eleZCoord, MatrixXi elementEigen)
{
    PetscErrorCode ierr;

    //Create vector for von-Mises stress
    ierr = VecDuplicate(XPhys, &VMS); CHKERRQ(ierr);
    PetscObjectSetName((PetscObject)VMS, "von-mises stress");
    VecZeroEntries(VMS);

    //Scatter U
    Vec U_SEQ;
    VecScatter utx;
    ierr = VecScatterCreateToAll(U, &utx, &U_SEQ); CHKERRQ(ierr);
    ierr = VecScatterBegin(utx, U, U_SEQ, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(utx, U, U_SEQ, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

    //Pointer to the displacement vector
    PetscScalar* up;
    ierr = VecGetArray(U_SEQ, &up);
    CHKERRQ(ierr);
    //Pointer to the von-Mises stress vector
    PetscScalar* vp;
    ierr = VecGetArray(VMS, &vp);
    CHKERRQ(ierr);
    //Pointer to the design variable vector
    PetscScalar* xp;
    ierr = VecGetArray(XPhys, &xp);
    CHKERRQ(ierr);

    PetscInt ele;
    VectorXd u(24), us(24);
    VectorXd esm(6), es(6);

    eleCoordStress.resize(8, 3);

    MatrixXd zero(1, 8);
    zero.fill(0);

    //Elasticity Matrix D
    double lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = 1.0 / (2.0 * (1.0 + nu));
    MatrixXd D(6, 6);

    D << lambda + 2.0 * mu, lambda, lambda, 0.0, 0.0, 0.0, lambda, lambda + 2.0 * mu, lambda, 0.0, 0.0, 0.0,
        lambda, lambda, lambda + 2.0 * mu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, mu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu;

    MatrixXd  Ndiff(3, 8), Jacob(3, 3), dN(3, 8), B(6, 24);

    PetscInt ldim;
    ierr = VecGetLocalSize(XPhys, &ldim); CHKERRQ(ierr);

    for (PetscInt ee = 0; ee < ldim; ee++)
    {
        u.fill(0);
        us.fill(0);

        for (PetscInt j = 0; j < COL_2; j++)
        {
            ele = elementEigen(ee, j);
            //Get the element displacement vector with the format [u1 v1 w1 ... u8 v8 w8]
            u(j * 3) = up[ele * 3];
            u(j * 3 + 1) = up[ele * 3 + 1];
            u(j * 3 + 2) = up[ele * 3 + 2];
        }

        //Transform element displacement vector to [u1 u2 ... u8 v1 v2 .. v8 w1 w2 ... w8]
        us(0) = u(0), us(1) = u(3), us(2) = u(6), us(3) = u(9), us(4) = u(12), us(5) = u(15), us(6) = u(18), us(7) = u(21),//[u1 u2 ... u8]
            us(8) = u(1), us(9) = u(4), us(10) = u(7), us(11) = u(10), us(12) = u(13), us(13) = u(16), us(14) = u(19), us(15) = u(22),//[v1 v2 ... v8]
            us(16) = u(2), us(17) = u(5), us(18) = u(8), us(19) = u(11), us(20) = u(14), us(21) = u(17), us(22) = u(20), us(23) = u(23);//[w1 w2 ... w8]

        for (int w = 0; w < 24; w++)
        {
            eleu(ee, w) = us(w);
        }

        eleCoordStress << eleXCoord.row(ee).transpose(), eleYCoord.row(ee).transpose(), eleZCoord.row(ee).transpose();

        //Calculate the isoparametric center point stress
        double xi = 0;
        double eta = 0;
        double zeta = 0;

        //Partial derivatives of shape functions
        Ndiff << -0.125 * (zeta - 1) * (eta - 1), 0.125 * (zeta - 1) * (eta - 1), 0.125 * (zeta - 1) * (eta + 1), -0.125 * (zeta - 1) * (eta + 1), 0.125 * (zeta + 1) * (eta - 1), -0.125 * (zeta + 1) * (eta - 1),
            -0.125 * (zeta + 1) * (eta + 1), 0.125 * (zeta + 1) * (eta + 1),//row(0)
            -0.125 * (zeta - 1) * (xi - 1), 0.125 * (zeta - 1) * (xi + 1), 0.125 * (zeta - 1) * (xi - 1), -0.125 * (zeta - 1) * (xi + 1), 0.125 * (zeta + 1) * (xi - 1), -0.125 * (zeta + 1) * (xi + 1),
            -0.125 * (zeta + 1) * (xi - 1), 0.125 * (zeta + 1) * (xi + 1),//row(1)
            -0.125 * (xi - 1) * (eta - 1), 0.125 * (xi + 1) * (eta - 1), 0.125 * (xi - 1) * (eta + 1), -0.125 * (xi + 1) * (eta + 1), 0.125 * (xi - 1) * (eta - 1), -0.125 * (xi + 1) * (eta - 1),
            -0.125 * (xi - 1) * (eta + 1), 0.125 * (xi + 1) * (eta + 1);//row(2)
        //Jacob 3*3
        Jacob = Ndiff * eleCoordStress;
        //dN 3*8
        dN = Jacob.inverse() * Ndiff;
        //B 6*24
        B << dN.row(0), zero, zero, zero, dN.row(1), zero, zero, zero, dN.row(2), dN.row(1), dN.row(0), zero, zero, dN.row(2), dN.row(1), dN.row(2), zero, dN.row(0);
        //element stress
        esm = Emax * D * B * us;
        //Stress relaxation
        es = PetscPowScalar(xp[ee], 0.5) * esm;

        //Calculate von-mises stress
        vp[ee] = (1 / sqrt(2)) * sqrt(pow(es(0) - es(1), 2) + pow(es(0) - es(2), 2) + pow(es(1) - es(2), 2) + 6 * (pow(es(3), 2) + pow(es(4), 2) + pow(es(5), 2)));
    }

    VecScatterDestroy(&utx);
    VecRestoreArray(U_SEQ, &up);
    VecRestoreArray(VMS, &vp);
    VecRestoreArray(XPhys, &xp);
    VecDestroy(&U_SEQ);

    return ierr;
}