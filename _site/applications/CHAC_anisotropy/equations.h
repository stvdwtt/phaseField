// List of variables and residual equations for the coupled Allen-Cahn/Cahn-Hilliard with anisotropic interfacial energy example application

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
	set_need_hessian				(0,false);

	set_need_value_residual_term	(0,true);
	set_need_gradient_residual_term	(0,true);

    // Variable 1
	set_variable_name				(1,"n");
	set_variable_type				(1,SCALAR);
	set_variable_equation_type		(1,PARABOLIC);

	set_need_value					(1,true);
	set_need_gradient				(1,true);
	set_need_hessian				(1,false);

	set_need_value_residual_term	(1,true);
	set_need_gradient_residual_term	(1,true);
}

// =================================================================================
// residualRHS
// =================================================================================
// This function calculates the residual equations for each variable. It takes
// "variable_list" as an input, which is a list of the value and derivatives of
// each of the variables at a specific quadrature point. The (x,y,z) location of
// that quadrature point is given by "q_point_loc". The function outputs residuals
// to variable_list. The index for each variable in this list corresponds to
// the index given at the top of this file.

template <int dim, int degree>
void customPDE<dim,degree>::residualRHS(variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
				 dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const {

// Concentration
scalarvalueType c = variable_list.get_scalar_value(0);
scalargradType cx = variable_list.get_scalar_gradient(0);

// Order parameter
scalarvalueType n = variable_list.get_scalar_value(1);
scalargradType nx = variable_list.get_scalar_gradient(1);

// Bulk terms
scalarvalueType faV = 0.5*c*c/16.0;
scalarvalueType facV = 0.5*c/8.0;
scalarvalueType faccV = constV(0.5/8.0);
scalarvalueType fbV = 0.5*(c-1.0)*(c-1.0)/16.0;
scalarvalueType fbcV = 0.5*(c-1.0)/8.0;
scalarvalueType fbccV = constV(0.5/8.0);
scalarvalueType hV = 3.0*n*n-2.0*n*n*n;
scalarvalueType hnV = 6.0*n-6.0*n*n;

// Calculation of interface normal vector
scalarvalueType normgradn = std::sqrt(nx.norm_square());
scalargradType normal = nx/(normgradn+constV(1.0e-16));

// Calculation of anisotropy gamma
scalarvalueType gamma;
if (dim == 2){
    gamma = 1.0+epsilonM*(4.0*(normal[0]*normal[0]*normal[0]*normal[0]+normal[1]*normal[1]*normal[1]*normal[1])-3.0);
}
else {
    gamma = 1.0+epsilonM*(4.0*(normal[0]*normal[0]*normal[0]*normal[0]+normal[1]*normal[1]*normal[1]*normal[1]+normal[2]*normal[2]*normal[2]*normal[2])-3.0);
}

// Derivatives of gamma with respect to the components of the unit normal
scalargradType dgammadnorm;
dgammadnorm[0] = (epsilonM*16.0*normal[0]*normal[0]*normal[0]);
dgammadnorm[1] = (epsilonM*16.0*normal[1]*normal[1]*normal[1]);
if (dim == 3){
    dgammadnorm[2] = (epsilonM*16.0*normal[2]*normal[2]*normal[2]);
}

// Product of projection matrix and dgammadnorm vector
scalargradType aniso;
for (unsigned int i=0; i<dim; ++i){
  for (unsigned int j=0; j<dim; ++j){
      aniso[i] += -normal[i]*normal[j]*dgammadnorm[j];
      if (i==j) aniso[i] +=dgammadnorm[j];
  }
}
// Anisotropic gradient term (see derivation)
aniso = gamma*(aniso*normgradn+gamma*nx);

// Residual expressions
scalarvalueType rcV = c;
scalargradType rcxV = constV(-McV*userInputs.dtValue)*(cx*((1.0-hV)*faccV+hV*fbccV)+nx*hnV*(fbcV-facV));
scalarvalueType rnV = n-constV(userInputs.dtValue*MnV)*(fbV-faV)*hnV;
scalargradType rnxV = constV(userInputs.dtValue*MnV)*(-aniso);

// Submission of residuals
variable_list.set_scalar_value_residual_term(0,rcV);
variable_list.set_scalar_gradient_residual_term(0,rcxV);
variable_list.set_scalar_value_residual_term(1,rnV);
variable_list.set_scalar_gradient_residual_term(1,rnxV);

}

// =================================================================================
// residualLHS (needed only if at least one equation is elliptic)
// =================================================================================
// This function calculates the residual equations for the iterative solver for
// elliptic equations.for each variable. It takes "variable_list" as an input,
// which is a list of the value and derivatives of each of the variables at a
// specific quadrature point. The (x,y,z) location of that quadrature point is given
// by "q_point_loc". The function outputs residual terms to "variable_list"
// for the left-hand-side of the residual equation for the iterative solver. The
// index for each variable in this list corresponds to
// the index given at the top of this file. If there are multiple elliptic equations,
// conditional statements should be used to ensure that the correct residual is
// being submitted. The index of the field being solved can be accessed by
// "this->currentFieldIndex".

template <int dim, int degree>
void customPDE<dim,degree>::residualLHS(variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
		dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const {
}
