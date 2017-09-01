// List of variables and residual equations for the coupled Allen-Cahn/Cahn-Hilliard example application

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

//c
scalarvalueType c = variable_list.get_scalar_value(0);
scalargradType cx = variable_list.get_scalar_gradient(0);

//n
scalarvalueType n = variable_list.get_scalar_value(1);
scalargradType nx = variable_list.get_scalar_gradient(1);

// Free energy for each phase and their first and second derivatives
scalarvalueType faV = (-1.6704-4.776*c+5.1622*c*c-2.7375*c*c*c+1.3687*c*c*c*c);
scalarvalueType facV = (-4.776 + 10.3244*c - 8.2125*c*c + 5.4748*c*c*c);
scalarvalueType faccV = (10.3244-16.425*c+16.4244*c*c);
scalarvalueType fbV = (5.0*c*c-5.9746*c-1.5924);
scalarvalueType fbcV = (10.0*c-5.9746);
scalarvalueType fbccV = constV(10.0);

// Interpolation function and its derivative
scalarvalueType hV = (10.0*n*n*n-15.0*n*n*n*n+6.0*n*n*n*n*n);
scalarvalueType hnV = (30.0*n*n-60.0*n*n*n+30.0*n*n*n*n);

// Residual equations
scalargradType muxV = ( cx*((1.0-hV)*faccV+hV*fbccV) + nx*((fbcV-facV)*hnV) );
scalarvalueType rcV = c;
scalargradType rcxV = (constV(-McV*userInputs.dtValue)*muxV);
scalarvalueType rnV = (n-constV(userInputs.dtValue*MnV)*(fbV-faV)*hnV);
scalargradType rnxV = (constV(-userInputs.dtValue*KnV*MnV)*nx);

// Residuals for the equation to evolve the concentration 
variable_list.set_scalar_value_residual_term(0,rcV);
variable_list.set_scalar_gradient_residual_term(0,rcxV);

// Residuals for the equation to evolve the order parameter
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
