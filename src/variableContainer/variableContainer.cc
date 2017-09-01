// All of the methods for the 'variableContainer' class
#include "../../include/variableContainer.h"

template <int dim, int degree, typename T>
variableContainer<dim,degree,T>::variableContainer(const dealii::MatrixFree<dim,double> &data, std::vector<variable_info> _varInfoList)
{
    varInfoList = _varInfoList;

    num_var = varInfoList.size();

    for (unsigned int i=0; i < num_var; i++){
        if (varInfoList[i].var_needed){
            if (varInfoList[i].is_scalar){
                dealii::FEEvaluation<dim,degree,degree+1,1,double> var(data, i);
                scalar_vars.push_back(var);
            }
            else {
                dealii::FEEvaluation<dim,degree,degree+1,dim,double> var(data, i);
                vector_vars.push_back(var);
            }
        }
    }

    // Generate lists of residual types and indices, and reserve space in the temporary vectors
    unsigned int num_scalar_values = 0;
    unsigned int num_scalar_gradients = 0;
    unsigned int num_vector_values = 0;
    unsigned int num_vector_gradients = 0;

    for (unsigned int var=0; var < num_var; var++){
        if (varInfoList[var].is_scalar){
            if (varInfoList[var].value_residual){
                scalar_value_index.push_back(num_scalar_values);
                num_scalar_values++;
            }
            else {
                scalar_value_index.push_back(-1);
            }
            if (varInfoList[var].gradient_residual){
                scalar_gradient_index.push_back(num_scalar_gradients);
                num_scalar_gradients++;
            }
            else {
                scalar_gradient_index.push_back(-1);
            }
            vector_value_index.push_back(-1);
            vector_gradient_index.push_back(-1);
        }
        else {
            if (varInfoList[var].value_residual){
                vector_value_index.push_back(num_vector_values);
                num_vector_values++;
            }
            else {
                vector_value_index.push_back(-1);
            }
            if (varInfoList[var].gradient_residual){
                vector_gradient_index.push_back(num_vector_gradients);
                num_vector_gradients++;
            }
            else {
                vector_gradient_index.push_back(-1);
            }
            scalar_value_index.push_back(-1);
            scalar_gradient_index.push_back(-1);
        }
    }

    scalar_value.reserve(num_scalar_values);
    scalar_gradient.reserve(num_scalar_gradients);
    vector_value.reserve(num_vector_values);
    vector_gradient.reserve(num_vector_gradients);

}

// Variant of the constructor where it reads from a fixed index of "data", used for post-processing
template <int dim, int degree, typename T>
variableContainer<dim,degree,T>::variableContainer(const dealii::MatrixFree<dim,double> &data, std::vector<variable_info> _varInfoList, unsigned int fixed_index)
{
    varInfoList = _varInfoList;

    num_var = varInfoList.size();

    for (unsigned int i=0; i < num_var; i++){
        if (varInfoList[i].var_needed){
            if (varInfoList[i].is_scalar){
                dealii::FEEvaluation<dim,degree,degree+1,1,double> var(data, fixed_index);
                scalar_vars.push_back(var);
            }
            else {
                dealii::FEEvaluation<dim,degree,degree+1,dim,double> var(data, fixed_index);
                vector_vars.push_back(var);
            }
        }
    }

    // Generate lists of residual types and indices, and reserve space in the temporary vectors
    unsigned int num_scalar_values = 0;
    unsigned int num_scalar_gradients = 0;
    unsigned int num_vector_values = 0;
    unsigned int num_vector_gradients = 0;

    for (unsigned int var=0; var < num_var; var++){
        if (varInfoList[var].is_scalar){
            if (varInfoList[var].value_residual){
                scalar_value_index.push_back(num_scalar_values);
                num_scalar_values++;
            }
            else {
                scalar_value_index.push_back(-1);
            }
            if (varInfoList[var].gradient_residual){
                scalar_gradient_index.push_back(num_scalar_gradients);
                num_scalar_gradients++;
            }
            else {
                scalar_gradient_index.push_back(-1);
            }
            vector_value_index.push_back(-1);
            vector_gradient_index.push_back(-1);
        }
        else {
            if (varInfoList[var].value_residual){
                vector_value_index.push_back(num_vector_values);
                num_vector_values++;
            }
            else {
                vector_value_index.push_back(-1);
            }
            if (varInfoList[var].gradient_residual){
                vector_gradient_index.push_back(num_vector_gradients);
                num_vector_gradients++;
            }
            else {
                vector_gradient_index.push_back(-1);
            }
            scalar_value_index.push_back(-1);
            scalar_gradient_index.push_back(-1);
        }
    }

    scalar_value.reserve(num_scalar_values);
    scalar_gradient.reserve(num_scalar_gradients);
    vector_value.reserve(num_vector_values);
    vector_gradient.reserve(num_vector_gradients);

}

template <int dim, int degree, typename T>
void variableContainer<dim,degree,T>::get_JxW(dealii::AlignedVector<T> & JxW){
    if (scalar_vars.size() > 0){
        scalar_vars[0].fill_JxW_values(JxW);
    }
    else {
        vector_vars[0].fill_JxW_values(JxW);
    }
}

template <int dim, int degree, typename T>
unsigned int variableContainer<dim,degree,T>::get_num_q_points(){
    if (scalar_vars.size() > 0){
        return scalar_vars[0].n_q_points;
    }
    else {
        return vector_vars[0].n_q_points;
    }
}

template <int dim, int degree, typename T>
dealii::Point<dim, T> variableContainer<dim,degree,T>::get_q_point_location(){
    if (scalar_vars.size() > 0){
        return scalar_vars[0].quadrature_point(q_point);
    }
    else {
        return vector_vars[0].quadrature_point(q_point);
    }
}

template <int dim, int degree, typename T>
void variableContainer<dim,degree,T>::reinit_and_eval(const std::vector<vectorType*> &src, unsigned int cell){

    for (unsigned int i=0; i<num_var; i++){
        if (varInfoList[i].var_needed){
            if (varInfoList[i].is_scalar) {
                scalar_vars[varInfoList[i].scalar_or_vector_index].reinit(cell);
                scalar_vars[varInfoList[i].scalar_or_vector_index].read_dof_values(*src[i]);
                scalar_vars[varInfoList[i].scalar_or_vector_index].evaluate(varInfoList[i].need_value, varInfoList[i].need_gradient, varInfoList[i].need_hessian);
            }
            else {
                vector_vars[varInfoList[i].scalar_or_vector_index].reinit(cell);
                vector_vars[varInfoList[i].scalar_or_vector_index].read_dof_values(*src[i]);
                vector_vars[varInfoList[i].scalar_or_vector_index].evaluate(varInfoList[i].need_value, varInfoList[i].need_gradient, varInfoList[i].need_hessian);
            }
        }
    }
}

template <int dim, int degree, typename T>
void variableContainer<dim,degree,T>::reinit_and_eval_LHS(const vectorType &src, const std::vector<vectorType*> solutionSet, unsigned int cell, unsigned int var_being_solved){

    for (unsigned int i=0; i<num_var; i++){
        if (varInfoList[i].var_needed){
            if (varInfoList[i].is_scalar) {
                scalar_vars[varInfoList[i].scalar_or_vector_index].reinit(cell);
                if (i == var_being_solved ){
                    scalar_vars[varInfoList[i].scalar_or_vector_index].read_dof_values(src);
                }
                else{
                    scalar_vars[varInfoList[i].scalar_or_vector_index].read_dof_values(*solutionSet[i]);
                }
                scalar_vars[varInfoList[i].scalar_or_vector_index].evaluate(varInfoList[i].need_value, varInfoList[i].need_gradient, varInfoList[i].need_hessian);
            }
            else {
                vector_vars[varInfoList[i].scalar_or_vector_index].reinit(cell);
                if (i == var_being_solved){
                    vector_vars[varInfoList[i].scalar_or_vector_index].read_dof_values(src);
                }
                else {
                    vector_vars[varInfoList[i].scalar_or_vector_index].read_dof_values(*solutionSet[i]);
                }
                vector_vars[varInfoList[i].scalar_or_vector_index].evaluate(varInfoList[i].need_value, varInfoList[i].need_gradient, varInfoList[i].need_hessian);
            }
        }
    }

}

template <int dim, int degree, typename T>
void variableContainer<dim,degree,T>::reinit(unsigned int cell){

    for (unsigned int i=0; i<num_var; i++){
        if (varInfoList[i].var_needed){
            if (varInfoList[i].is_scalar) {
                scalar_vars[varInfoList[i].scalar_or_vector_index].reinit(cell);
            }
            else {
                vector_vars[varInfoList[i].scalar_or_vector_index].reinit(cell);
            }
        }
    }
}


template <int dim, int degree, typename T>
void variableContainer<dim,degree,T>::integrate_and_distribute(std::vector<vectorType*> &dst){

    for (unsigned int i=0; i<num_var; i++){
        if (varInfoList[i].is_scalar) {
            scalar_vars[varInfoList[i].scalar_or_vector_index].integrate(varInfoList[i].value_residual, varInfoList[i].gradient_residual);
            scalar_vars[varInfoList[i].scalar_or_vector_index].distribute_local_to_global(*dst[i]);
        }
        else {
            vector_vars[varInfoList[i].scalar_or_vector_index].integrate(varInfoList[i].value_residual, varInfoList[i].gradient_residual);
            vector_vars[varInfoList[i].scalar_or_vector_index].distribute_local_to_global(*dst[i]);
        }
    }

}

template <int dim, int degree, typename T>
void variableContainer<dim,degree,T>::integrate_and_distribute_LHS(vectorType &dst, unsigned int var_being_solved){

    //integrate
    if (varInfoList[var_being_solved].is_scalar) {
    	scalar_vars[varInfoList[var_being_solved].scalar_or_vector_index].integrate(varInfoList[var_being_solved].value_residual, varInfoList[var_being_solved].gradient_residual);
    	scalar_vars[varInfoList[var_being_solved].scalar_or_vector_index].distribute_local_to_global(dst);
    }
    else {
    	vector_vars[varInfoList[var_being_solved].scalar_or_vector_index].integrate(varInfoList[var_being_solved].value_residual, varInfoList[var_being_solved].gradient_residual);
    	vector_vars[varInfoList[var_being_solved].scalar_or_vector_index].distribute_local_to_global(dst);
    }
}

// Need to add index checking to these functions so that an error is thrown if the index wasn't set
template <int dim, int degree, typename T>
T variableContainer<dim,degree,T>::get_scalar_value(unsigned int global_variable_index) const
{
    if (varInfoList[global_variable_index].need_value){
        return scalar_vars[varInfoList[global_variable_index].scalar_or_vector_index].get_value(q_point);
    }
    else {
        std::cerr << "PRISMS-PF Error: Attempted access of a variable value that was not marked as needed in 'parameters.in'. Double-check the indices in user functions where a variable value is requested." << std::endl;
        abort();
    }
}

template <int dim, int degree, typename T>
dealii::Tensor<1, dim, T > variableContainer<dim,degree,T>::get_scalar_gradient(unsigned int global_variable_index) const
{
    if (varInfoList[global_variable_index].need_gradient){
        return scalar_vars[varInfoList[global_variable_index].scalar_or_vector_index].get_gradient(q_point);
    }
    else {
        std::cerr << "PRISMS-PF Error: Attempted access of a variable value that was not marked as needed in 'parameters.in'. Double-check the indices in user functions where a variable value is requested." << std::endl;
        abort();
    }
}

template <int dim, int degree, typename T>
dealii::Tensor<2, dim, T > variableContainer<dim,degree,T>::get_scalar_hessian(unsigned int global_variable_index) const
{
    if (varInfoList[global_variable_index].need_hessian){
        return scalar_vars[varInfoList[global_variable_index].scalar_or_vector_index].get_hessian(q_point);
    }
    else {
        std::cerr << "PRISMS-PF Error: Attempted access of a variable value that was not marked as needed in 'parameters.in'. Double-check the indices in user functions where a variable value is requested." << std::endl;
        abort();
    }
}

template <int dim, int degree, typename T>
dealii::Tensor<1, dim, T > variableContainer<dim,degree,T>::get_vector_value(unsigned int global_variable_index) const
{
    if (varInfoList[global_variable_index].need_value){
        return vector_vars[varInfoList[global_variable_index].scalar_or_vector_index].get_value(q_point);
    }
    else {
        std::cerr << "PRISMS-PF Error: Attempted access of a variable value that was not marked as needed in 'parameters.in'. Double-check the indices in user functions where a variable value is requested." << std::endl;
        abort();
    }
}

template <int dim, int degree, typename T>
dealii::Tensor<2, dim, T > variableContainer<dim,degree,T>::get_vector_gradient(unsigned int global_variable_index) const
{
    if (varInfoList[global_variable_index].need_gradient){
        return vector_vars[varInfoList[global_variable_index].scalar_or_vector_index].get_gradient(q_point);
    }
    else {
        std::cerr << "PRISMS-PF Error: Attempted access of a variable value that was not marked as needed in 'parameters.in'. Double-check the indices in user functions where a variable value is requested." << std::endl;
        abort();
    }
}

template <int dim, int degree, typename T>
dealii::Tensor<3, dim, T > variableContainer<dim,degree,T>::get_vector_hessian(unsigned int global_variable_index) const
{
    if (varInfoList[global_variable_index].need_hessian){
        return vector_vars[varInfoList[global_variable_index].scalar_or_vector_index].get_hessian(q_point);
    }
    else {
        std::cerr << "PRISMS-PF Error: Attempted access of a variable value that was not marked as needed in 'parameters.in'. Double-check the indices in user functions where a variable value is requested." << std::endl;
        abort();
    }
}

template <int dim, int degree, typename T>
void variableContainer<dim,degree,T>::set_scalar_value_residual_term(unsigned int global_variable_index, T val){
    scalar_vars[varInfoList[global_variable_index].scalar_or_vector_index].submit_value(val,q_point);

    // Hold-over from previous version that caches the residual terms instead of over-writing what's in scalar_vars.
    // This approach was abandoned because it caused a 20% decrease in speed.
    //scalar_value[scalar_value_index[global_variable_index]] = val;
}

template <int dim, int degree, typename T>
void variableContainer<dim,degree,T>::set_scalar_gradient_residual_term(unsigned int global_variable_index, dealii::Tensor<1, dim, T > grad){
    scalar_vars[varInfoList[global_variable_index].scalar_or_vector_index].submit_gradient(grad,q_point);

    // Hold-over from previous version that caches the residual terms instead of over-writing what's in scalar_vars.
    // This approach was abandoned because it caused a 20% decrease in speed.
    //scalar_gradient[scalar_gradient_index[global_variable_index]] = grad;
}

template <int dim, int degree, typename T>
void variableContainer<dim,degree,T>::set_vector_value_residual_term(unsigned int global_variable_index, dealii::Tensor<1, dim, T > val){
    vector_vars[varInfoList[global_variable_index].scalar_or_vector_index].submit_value(val,q_point);

    // Hold-over from previous version that caches the residual terms instead of over-writing what's in vector_vars.
    // This approach was abandoned because it caused a 20% decrease in speed.
    //vector_value[vector_value_index[global_variable_index]] = val;
}
template <int dim, int degree, typename T>
void variableContainer<dim,degree,T>::set_vector_gradient_residual_term(unsigned int global_variable_index, dealii::Tensor<2, dim, T > grad){
    vector_vars[varInfoList[global_variable_index].scalar_or_vector_index].submit_gradient(grad,q_point);

    // Hold-over from previous version that caches the residual terms instead of over-writing what's in vector_vars.
    // This approach was abandoned because it caused a 20% decrease in speed.
    //vector_gradient[vector_gradient_index[global_variable_index]] = grad;
}


// Hold-over from previous version that caches the residual terms instead of over-writing what's in vector_vars.
// This approach was abandoned because it caused a 20% decrease in speed.
// template <int dim, int degree, typename T>
// void variableContainer<dim,degree,T>::apply_residuals(){
//
//     for (unsigned int i=0; i<num_var; i++){
//         if (varInfoList[i].is_scalar){
//             if (varInfoList[i].value_residual){
//                 scalar_vars[varInfoList[i].scalar_or_vector_index].submit_value(scalar_value[scalar_value_index[i]],q_point);
//             }
//             if (varInfoList[i].gradient_residual){
//                 scalar_vars[varInfoList[i].scalar_or_vector_index].submit_gradient(scalar_gradient[scalar_gradient_index[i]],q_point);
//             }
//         }
//         else {
//             if (varInfoList[i].value_residual){
//                 vector_vars[varInfoList[i].scalar_or_vector_index].submit_value(vector_value[vector_value_index[i]],q_point);
//             }
//             if (varInfoList[i].gradient_residual){
//                 vector_vars[varInfoList[i].scalar_or_vector_index].submit_gradient(vector_gradient[vector_gradient_index[i]],q_point);
//             }
//         }
//     }
// }

template class variableContainer<2,1,dealii::VectorizedArray<double> >;
template class variableContainer<2,2,dealii::VectorizedArray<double> >;
template class variableContainer<2,3,dealii::VectorizedArray<double> >;
template class variableContainer<3,1,dealii::VectorizedArray<double> >;
template class variableContainer<3,2,dealii::VectorizedArray<double> >;
template class variableContainer<3,3,dealii::VectorizedArray<double> >;
