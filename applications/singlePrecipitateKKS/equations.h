// List of variables and residual equations for the Precipitate Evolution example application

// =================================================================================
// Set the attributes of the primary field variables
// =================================================================================
void variableAttributeLoader::loadVariableAttributes(){
	// Variable 0
	set_variable_name				(0,"c");
	set_variable_type				(0,SCALAR);
	set_variable_equation_type		(0,PARABOLIC);

	set_need_value					(0,true);
	set_need_gradient				(0,true);
	set_need_hessian					(0,false);

	set_need_value_residual_term		(0,true);
	set_need_gradient_residual_term	(0,true);

	set_need_value_LHS				(0,false);
	set_need_gradient_LHS			(0,false);
	set_need_hessian_LHS				(0,false);
	set_need_value_residual_term_LHS		(0,false);
	set_need_gradient_residual_term_LHS	(0,false);

	// Variable 1
	set_variable_name				(1,"n1");
	set_variable_type				(1,SCALAR);
	set_variable_equation_type		(1,PARABOLIC);

	set_need_value					(1,true);
	set_need_gradient				(1,true);
	set_need_hessian					(1,false);

	set_need_value_residual_term		(1,true);
	set_need_gradient_residual_term	(1,true);

	set_need_value_LHS				(1,true);
	set_need_gradient_LHS			(1,false);
	set_need_hessian_LHS				(1,false);
	set_need_value_residual_term_LHS		(1,false);
	set_need_gradient_residual_term_LHS	(1,false);

	// Variable 2
	set_variable_name				(2,"u");
	set_variable_type				(2,VECTOR);
	set_variable_equation_type		(2,ELLIPTIC);

	set_need_value					(2,false);
	set_need_gradient				(2,true);
	set_need_hessian					(2,false);

	set_need_value_residual_term		(2,false);
	set_need_gradient_residual_term	(2,true);

	set_need_value_LHS				(2,false);
	set_need_gradient_LHS			(2,true);
	set_need_hessian_LHS				(2,false);
	set_need_value_residual_term_LHS		(2,false);
	set_need_gradient_residual_term_LHS	(2,true);

}

// =================================================================================
// residualRHS
// =================================================================================
// This function calculates the residual equations for each variable. It takes
// "modelVariablesList" as an input, which is a list of the value and derivatives of
// each of the variables at a specific quadrature point. The (x,y,z) location of
// that quadrature point is given by "q_point_loc". The function outputs
// "modelResidualsList", a list of the value and gradient terms of the residual for
// each residual equation. The index for each variable in these lists corresponds to
// the order it is defined at the top of this file (starting at 0).
template <int dim, int degree>
void customPDE<dim,degree>::residualRHS(variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
												dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const {

// The concentration and its derivatives 
scalarvalueType c = variable_list.get_scalar_value(0);
scalargradType cx = variable_list.get_scalar_gradient(0);

// The first order parameter and its derivatives 
scalarvalueType n1 = variable_list.get_scalar_value(1);
scalargradType n1x = variable_list.get_scalar_gradient(1);

// The derivative of the displacement vector 
vectorgradType ux = variable_list.get_vector_gradient(2);
vectorgradType ruxV;

vectorhessType uxx;

bool c_dependent_misfit = false;
for (unsigned int i=0; i<dim; i++){
	for (unsigned int j=0; j<dim; j++){
		if (std::abs(sfts_linear1[i][j])>1.0e-12){
			c_dependent_misfit = true;
		}
	}
}

if (c_dependent_misfit == true){
	uxx = variable_list.get_vector_hessian(2);
}

// Interpolation functions
scalarvalueType h1V = (3.0*n1*n1-2.0*n1*n1*n1);
scalarvalueType hn1V = (6.0*n1-6.0*n1*n1);

// Calculate c_alpha and c_beta from c
scalarvalueType c_alpha = ((B2*c+0.5*(B1-A1)*h1V)/(A2*h1V+B2*(1.0-h1V)));
scalarvalueType c_beta = ((A2*c+0.5*(A1-B1)*(1.0-h1V))/(A2*h1V+B2*(1.0-h1V)));

// Free energy functions
scalarvalueType faV = (A2*c_alpha*c_alpha + A1*c_alpha + A0);
scalarvalueType facV = (2.0*A2*c_alpha +A1);
scalarvalueType faccV = constV(2.0*A2);
scalarvalueType fbV = (B2*c_beta*c_beta + B1*c_beta + B0);
scalarvalueType fbcV = (2.0*B2*c_beta +B1);
scalarvalueType fbccV = constV(2.0*B2);

// This double-well function can be used to tune the interfacial energy
scalarvalueType fbarrierV = (n1*n1-2.0*n1*n1*n1+n1*n1*n1*n1);
scalarvalueType fbarriernV = (2.0*n1-6.0*n1*n1+4.0*n1*n1*n1);

// Calculate the derivatives of c_beta (derivatives of c_alpha aren't needed)
scalarvalueType cbnV, cbcV, cbcnV;

cbcV = faccV/( (constV(1.0)-h1V)*fbccV + h1V*faccV );
cbnV = hn1V * (c_alpha - c_beta) * cbcV;
cbcnV = (faccV * (fbccV-faccV) * hn1V)/( ((1.0-h1V)*fbccV + h1V*faccV)*((1.0-h1V)*fbccV + h1V*faccV) );  // Note: this is only true if faV and fbV are quadratic

// Calculate the stress-free transformation strain and its derivatives at the quadrature point
dealii::Tensor<2, dim, dealii::VectorizedArray<double> > sfts1, sfts1c, sfts1cc, sfts1n, sfts1cn;

for (unsigned int i=0; i<dim; i++){
for (unsigned int j=0; j<dim; j++){
	// Polynomial fits for the stress-free transformation strains, of the form: sfts = a_p * c_beta + b_p
	sfts1[i][j] = constV(sfts_linear1[i][j])*c_beta + constV(sfts_const1[i][j]);
	sfts1c[i][j] = constV(sfts_linear1[i][j]) * cbcV;
	sfts1cc[i][j] = constV(0.0);
	sfts1n[i][j] = constV(sfts_linear1[i][j]) * cbnV;
	sfts1cn[i][j] = constV(sfts_linear1[i][j]) * cbcnV;
}
}

//compute E2=(E-E0)
dealii::VectorizedArray<double> E2[dim][dim], S[dim][dim];

for (unsigned int i=0; i<dim; i++){
for (unsigned int j=0; j<dim; j++){
	  E2[i][j]= constV(0.5)*(ux[i][j]+ux[j][i])-( sfts1[i][j]*h1V );
}
}

//compute stress
//S=C*(E-E0)
// Compute stress tensor (which is equal to the residual, Rux)
dealii::VectorizedArray<double> CIJ_combined[CIJ_tensor_size][CIJ_tensor_size];

if (n_dependent_stiffness == true){
for (unsigned int i=0; i<2*dim-1+dim/3; i++){
	  for (unsigned int j=0; j<2*dim-1+dim/3; j++){
		  CIJ_combined[i][j] = CIJ_Mg[i][j]*(constV(1.0)-h1V) + CIJ_Beta[i][j]*h1V;
	  }
}
computeStress<dim>(CIJ_combined, E2, S);
}
else{
computeStress<dim>(CIJ_Mg, E2, S);
}


// Fill residual corresponding to mechanics
// R=-C*(E-E0)

for (unsigned int i=0; i<dim; i++){
for (unsigned int j=0; j<dim; j++){
	  ruxV[i][j] = - S[i][j];
}
}

// Compute one of the stress terms in the order parameter chemical potential, nDependentMisfitACp = -C*(E-E0)*(E0_n)
dealii::VectorizedArray<double> nDependentMisfitAC1=constV(0.0);

for (unsigned int i=0; i<dim; i++){
for (unsigned int j=0; j<dim; j++){
	  nDependentMisfitAC1+=-S[i][j]*(sfts1n[i][j]*h1V + sfts1[i][j]*hn1V);
}
}


// Compute the other stress term in the order parameter chemical potential, heterMechACp = 0.5*Hn*(C_beta-C_alpha)*(E-E0)*(E-E0)
dealii::VectorizedArray<double> heterMechAC1=constV(0.0);
dealii::VectorizedArray<double> S2[dim][dim];

if (n_dependent_stiffness == true){
	computeStress<dim>(CIJ_Beta-CIJ_Mg, E2, S2);

	for (unsigned int i=0; i<dim; i++){
		for (unsigned int j=0; j<dim; j++){
			heterMechAC1 += S2[i][j]*E2[i][j];
		}
	}
	heterMechAC1 = 0.5*hn1V*heterMechAC1;
}

// compute the stress term in the gradient of the concentration chemical potential, grad_mu_el = -[C*(E-E0)*E0c]x, must be a vector with length dim
scalargradType grad_mu_el;

if (c_dependent_misfit == true){
	dealii::VectorizedArray<double> E3[dim][dim], S3[dim][dim];

	for (unsigned int i=0; i<dim; i++){
		for (unsigned int j=0; j<dim; j++){
			E3[i][j] =  -( sfts1c[i][j]*h1V );
		}
	}

	if (n_dependent_stiffness == true){
		computeStress<dim>(CIJ_combined, E3, S3);
	}
	else{
		computeStress<dim>(CIJ_Mg, E3, S3);
	}

	for (unsigned int i=0; i<dim; i++){
		for (unsigned int j=0; j<dim; j++){
			for (unsigned int k=0; k<dim; k++){
				grad_mu_el[k] += S3[i][j] * (constV(0.5)*(uxx[i][j][k]+uxx[j][i][k]) + E3[i][j]*cx[k]
														  - ( (sfts1[i][j]*hn1V + sfts1n[i][j]*h1V) *n1x[k]) );

				grad_mu_el[k]+= - S[i][j] * ( (sfts1c[i][j]*hn1V + h1V*sfts1cn[i][j])*n1x[k] + ( sfts1cc[i][j]*h1V )*cx[k]);

				if (n_dependent_stiffness == true){
					grad_mu_el[k]+= S2[i][j] * (hn1V*n1x[k]) * (E3[i][j]);

				}
			}
		}
	}
}


//compute K*nx
scalargradType Knx1;
for (unsigned int a=0; a<dim; a++) {
Knx1[a]=0.0;
for (unsigned int b=0; b<dim; b++){
	  Knx1[a]+=constV(Kn1[a][b])*n1x[b];
}
}

// Define the residual expressions
scalarvalueType rcV = (c);
scalargradType rcxTemp = ( cx + n1x*(c_alpha-c_beta)*hn1V + grad_mu_el * (h1V*faccV+(constV(1.0)-h1V)*fbccV)/(faccV*fbccV) );
scalargradType rcxV = (constV(-userInputs.dtValue)*McV*rcxTemp);

scalarvalueType rn1V = (n1-constV(userInputs.dtValue*Mn1V)*( (fbV-faV)*hn1V - (c_beta-c_alpha)*facV*hn1V + W*fbarriernV + nDependentMisfitAC1 + heterMechAC1));
scalargradType rn1xV = (constV(-userInputs.dtValue*Mn1V)*Knx1);


// Set the residuals
variable_list.set_scalar_value_residual_term(0,rcV);
variable_list.set_scalar_gradient_residual_term(0,rcxV);

variable_list.set_scalar_value_residual_term(1,rn1V);
variable_list.set_scalar_gradient_residual_term(1,rn1xV);

variable_list.set_vector_gradient_residual_term(2,ruxV);

}

// =================================================================================
// residualLHS (needed only if at least one equation is elliptic)
// =================================================================================
// This function calculates the residual equations for the iterative solver for
// elliptic equations.for each variable. It takes "modelVariablesList" as an input,
// which is a list of the value and derivatives of each of the variables at a
// specific quadrature point. The (x,y,z) location of that quadrature point is given
// by "q_point_loc". The function outputs "modelRes", the value and gradient terms of
// for the left-hand-side of the residual equation for the iterative solver. The
// index for each variable in these lists corresponds to the order it is defined at
// the top of this file (starting at 0), not counting variables that have
// "need_val_LHS", "need_grad_LHS", and "need_hess_LHS" all set to "false". If there
// are multiple elliptic equations, conditional statements should be used to ensure
// that the correct residual is being submitted. The index of the field being solved
// can be accessed by "this->currentFieldIndex".
template <int dim, int degree>
void customPDE<dim,degree>::residualLHS(variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
		dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const {

// The first order parameter and its derivatives 
scalarvalueType n1 = variable_list.get_scalar_value(1);

// The derivative of the displacement vector 
vectorgradType ux = variable_list.get_vector_gradient(2);
vectorgradType ruxV;

// Interpolation function
scalarvalueType h1V = (3.0*n1*n1-2.0*n1*n1*n1);

// Take advantage of E being simply 0.5*(ux + transpose(ux)) and use the dealii "symmetrize" function
dealii::Tensor<2, dim, dealii::VectorizedArray<double> > E;
E = symmetrize(ux);

// Compute stress tensor (which is equal to the residual, Rux)
if (n_dependent_stiffness == true){
	dealii::Tensor<2, CIJ_tensor_size, dealii::VectorizedArray<double> > CIJ_combined;
	CIJ_combined = CIJ_Mg*(constV(1.0)-h1V);
	CIJ_combined += CIJ_Beta*(h1V);

	computeStress<dim>(CIJ_combined, E, ruxV);
}
else{
	computeStress<dim>(CIJ_Mg, E, ruxV);
}

variable_list.set_vector_gradient_residual_term(2,ruxV);

}
