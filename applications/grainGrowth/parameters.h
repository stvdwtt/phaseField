// Parameter list for the grain growth example problem.
// The model used for this example application is a simplified version of the model from:
// Gentry and Thornton, "Simulating Recrystalization in Titanium using the Phase Field Method",
// IOP Conference Series: Mater. Sci. Eng. 89 (2015) 012024.

// All strictly numerical parameters should be set in this file

// =================================================================================
// Set the number of dimensions (1, 2, or 3 for a 1D, 2D, or 3D calculation)
// =================================================================================
#define problemDIM 2

// =================================================================================
// Set the length of the domain in all three dimensions
// =================================================================================
// Each axes spans from zero to the specified length
#define spanX 100.0
#define spanY 100.0
#define spanZ 100.0

// =================================================================================
// Set the element parameters
// =================================================================================
// The number of elements in each direction is 2^(refineFactor) * subdivisions
// For optimal performance, use refineFactor primarily to determine the element size
#define subdivisionsX 1
#define subdivisionsY 1
#define subdivisionsZ 1
#define refineFactor 7

// Set the polynomial degree of the element (suggested values: 1 or 2)
#define finiteElementDegree 1

// =================================================================================
// Set the adaptive mesh refinement parameters
// =================================================================================
// Set the flag determining if adaptive meshing is activated
#define hAdaptivity true

// Set the maximum and minimum level of refinement
#define maxRefinementLevel (refineFactor)
#define minRefinementLevel (refineFactor-3)

// Set the fields used to determine the refinement. Fields determined by the order
// declared in "equations.h", starting at zero
#define refineCriterionFields {0,1,2,3,4,5,6,7,8,9}

// Set the maximum and minimum value of the fields where the mesh should be refined
#define refineWindowMax {0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999}
#define refineWindowMin {0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001}

// Set the number of time steps between remeshing operations
#define skipRemeshingSteps 1000

// =================================================================================
// Set the time step parameters
// =================================================================================
// The size of the time step
#define timeStep 1.0e-3

// The simulation ends when either timeFinal is reached or the number of time steps
// equals timeIncrements
#define timeFinal 20.0
#define timeIncrements 20000

// =================================================================================
// Set the output parameters
// =================================================================================
// Each field in the problem will be output is writeOutput is set to "true"
#define writeOutput true

// Type of spacing between outputs ("EQUAL_SPACING", "LOG_SPACING", "N_PER_DECADE",
// or "LIST")
#define outputCondition "EQUAL_SPACING"

// Number of times the program outputs the fields (total number for "EQUAL_SPACING"
// and "LOG_SPACING", number per decade for "N_PER_DECADE", ignored for "LIST")
#define numOutputs 10

// User-defined list of time steps where the program should output. Only used if
// outputCondition is "LIST"
#define outputList {}

// Status is printed to the screen every skipPrintSteps
#define skipPrintSteps 10

// =================================================================================
// Set the flag determining if the total free energy is calculated for each output
// =================================================================================
#define calcEnergy false







