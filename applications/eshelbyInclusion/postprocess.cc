// =============================================================================================
// loadPostProcessorVariableAttributes: Set the attributes of the postprocessing variables
// =============================================================================================
// This function is analogous to 'loadVariableAttributes' in 'equations.h', but for
// the postprocessing expressions. It sets the attributes for each postprocessing
// expression, including its name, whether it is a vector or scalar (only scalars are
// supported at present), its dependencies on other variables and their derivatives,
// and whether to calculate an integral of the postprocessed quantity over the entire
// domain. Note: this function is not a member of customPDE.

void variableAttributeLoader::loadPostProcessorVariableAttributes(){
	// Variable 0
	set_variable_name				(0,"f_tot");
	set_variable_type				(0,SCALAR);

    set_dependencies_value_term_RHS(0, "grad(u)");
    set_dependencies_gradient_term_RHS(0, "");

    set_output_integral         	(0,true);


}

// =============================================================================================
// postProcessedFields: Set the postprocessing expressions
// =============================================================================================
// This function is analogous to 'explicitEquationRHS' and 'nonExplicitEquationRHS' in
// equations.h. It takes in "variable_list" and "q_point_loc" as inputs and outputs two terms in
// the expression for the postprocessing variable -- one proportional to the test
// function and one proportional to the gradient of the test function. The index for
// each variable in this list corresponds to the index given at the top of this file (for
// submitting the terms) and the index in 'equations.h' for assigning the values/derivatives of
// the primary variables.

template <int dim,int degree>
void customPDE<dim,degree>::postProcessedFields(const variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
				variableContainer<dim,degree,dealii::VectorizedArray<double> > & pp_variable_list,
												const dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const {

    // --- Getting the values and derivatives of the model variables ---

		// The derivative of the displacement vector (names here should match those in the macros above)
		vectorgradType ux = variable_list.get_vector_gradient(1);

    // --- Setting the expressions for the terms in the postprocessing expressions ---
		scalarvalueType dist, phi;

		// Scaled distance from the center of the inclusion
		dist = std::sqrt((q_point_loc[0]-constV(center[0]))*(q_point_loc[0]-constV(center[0]))/semiaxes[0]/semiaxes[0]
							+(q_point_loc[1]-constV(center[1]))*(q_point_loc[1]-constV(center[1]))/semiaxes[1]/semiaxes[1]
							+(q_point_loc[2]-constV(center[2]))*(q_point_loc[2]-constV(center[2]))/semiaxes[2]/semiaxes[2]);

		phi = (0.5- 0.5*(1.0-std::exp(-20.0*(dist-constV(1.0))))/ (constV(1.0)+std::exp(-20.0*(dist-constV(1.0)))));

		for (unsigned int n=0; n < dist.n_array_elements; n++){
			if (dist[n] < 1.0){
				phi[n] = 1.0;
			}
			else{
				phi[n] = 0.0;
			}
		}


		//compute strain tensor
		dealii::VectorizedArray<double> E[dim][dim], S[dim][dim];
		for (unsigned int i=0; i<dim; i++){
			for (unsigned int j=0; j<dim; j++){
				E[i][j]= constV(0.5)*(ux[i][j]+ux[j][i])-sfts[i][j] * phi;
			}
		}

		//compute stress tensor
		computeStress<dim>(CIJ_matrix * (1.0 - phi) + CIJ_ppt * phi, E, S);


		scalarvalueType f_el = constV(0.0);

		for (unsigned int i=0; i<dim; i++){
			for (unsigned int j=0; j<dim; j++){
				f_el += constV(0.5) * S[i][j]*E[i][j];
			}
		}

// --- Submitting the terms for the postprocessing expressions ---

pp_variable_list.set_scalar_value_term_RHS(0, f_el);

}
