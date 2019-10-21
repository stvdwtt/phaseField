#include "../../include/matrixFreePDE.h"

template <int dim, int degree>
class customPDE: public MatrixFreePDE<dim,degree>
{
public:
    // Constructor
    customPDE(userInputParameters<dim> _userInputs): MatrixFreePDE<dim,degree>(_userInputs) , userInputs(_userInputs) {};

    // Function to set the initial conditions (in ICs_and_BCs.h)
    void setInitialCondition(const dealii::Point<dim> &p, const unsigned int index, double & scalar_IC, dealii::Vector<double> & vector_IC);

    // Function to set the non-uniform Dirichlet boundary conditions (in ICs_and_BCs.h)
    void setNonUniformDirichletBCs(const dealii::Point<dim> &p, const unsigned int index, const unsigned int direction, const double time, double & scalar_BC, dealii::Vector<double> & vector_BC);

private:
	#include "../../include/typeDefs.h"

	const userInputParameters<dim> userInputs;

	// Function to set the RHS of the governing equations for explicit time dependent equations (in equations.h)
    void explicitEquationRHS(variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
					 dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const;

    // Function to set the RHS of the governing equations for all other equations (in equations.h)
    void nonExplicitEquationRHS(variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
					 dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const;

	// Function to set the LHS of the governing equations (in equations.h)
	void equationLHS(variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
					 dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const;

	// Function to set postprocessing expressions (in postprocess.h)
	#ifdef POSTPROCESS_FILE_EXISTS
	void postProcessedFields(const variableContainer<dim,degree,dealii::VectorizedArray<double> > & variable_list,
					variableContainer<dim,degree,dealii::VectorizedArray<double> > & pp_variable_list,
					const dealii::Point<dim, dealii::VectorizedArray<double> > q_point_loc) const;
	#endif

	// Function to set the nucleation probability (in nucleation.h)
	#ifdef NUCLEATION_FILE_EXISTS
	double getNucleationProbability(variableValueContainer variable_value, double dV) const;
	#endif

	// ================================================================
	// Methods specific to this subclass
	// ================================================================

  void adaptiveRefineCriterion();

	// ================================================================
	// Model constants specific to this subclass
	// ================================================================

	const static unsigned int CIJ_tensor_size =2*dim-1+dim/3;
	dealii::Tensor<2,CIJ_tensor_size> CIJ_matrix = userInputs.get_model_constant_elasticity_tensor("CIJ_matrix");
  dealii::Tensor<2,CIJ_tensor_size> CIJ_ppt = userInputs.get_model_constant_elasticity_tensor("CIJ_ppt");

  dealii::Tensor<2,dim> sfts = userInputs.get_model_constant_rank_2_tensor("sfts");
  dealii::Tensor<1,dim> semiaxes = userInputs.get_model_constant_rank_1_tensor("semiaxes");
  dealii::Tensor<1,dim> center = userInputs.get_model_constant_rank_1_tensor("center");
  double refine_rad = userInputs.get_model_constant_double("refine_rad");

	// ================================================================

};

//Special implementation of adaptive mesh criterion to make sure grain boundary region is adapted to the highest level
template <int dim, int degree>
void customPDE<dim,degree>::adaptiveRefineCriterion(){

    //Custom defined estimation criterion

    std::vector<std::vector<double> > errorOutV;


    QGaussLobatto<dim>  quadrature(degree+1);
    FEValues<dim> fe_values (*(this->FESet[userInputs.refinement_criteria[0].variable_index]), quadrature, update_values|update_quadrature_points);
    const unsigned int num_quad_points = quadrature.size();
    std::vector<dealii::Point<dim> > q_point_list(num_quad_points);

    std::vector<double> errorOut(num_quad_points);

    typename DoFHandler<dim>::active_cell_iterator cell = this->dofHandlersSet_nonconst[0]->begin_active(), endc = this->dofHandlersSet_nonconst[0]->end();

    typename parallel::distributed::Triangulation<dim>::active_cell_iterator t_cell = this->triangulation.begin_active();

    for (;cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            fe_values.reinit (cell);

    		    q_point_list = fe_values.get_quadrature_points();

            bool mark_refine = false;

            for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
                  if ((q_point_list[q_point]-center).norm() < refine_rad){
                    mark_refine = true;
                  }
            }

            errorOutV.clear();

            //limit the maximal and minimal refinement depth of the mesh
            unsigned int current_level = t_cell->level();

            if ( (mark_refine && current_level < userInputs.max_refinement_level) ){
                cell->set_refine_flag();
            }
            else if (!mark_refine && current_level > userInputs.min_refinement_level) {
                cell->set_coarsen_flag();
            }

        }
        ++t_cell;
    }

}
