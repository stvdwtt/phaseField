PRISMS-PF
=================
<B> Useful Links:</B>

[Code repository](https://github.com/prisms-center/phaseField) <br>
[Doxygen Code documentation](https://goo.gl/00y23N) <br>
[User registration link](http://goo.gl/forms/GXo7Im8p2Y) <br>
[User forum](https://groups.google.com/forum/#!forum/prisms-pf-users) <br>
[Training slides/exercises](https://goo.gl/BBTkJ8)


<B>Version information:</B>

This version of the code, 2.0, contains substantial changes in the user interface from version 1.x. For information concerning the differences between versions, please consult version_changes.md.   

<B>What is PRISMS-PF?</B>

PRISMS-PF is a powerful, massively parallel finite element code for conducting phase field and other related simulations of microstructural evolution.  The phase field method is commonly used for predicting the evolution if microstructures under a wide range of conditions and material systems. PRISMS-PF provides a simple interface for solving customizable systems of partial differential equations of the type commonly found in phase field models, and has 11 pre-built application modules, including for precipitate evolution, grain growth, and spinodal decomposition.

With PRISMS-PF, you have access to adaptive meshing and parallelization with near-ideal scaling for over a thousand processors. Moreover, the matrix-free framework from the deal.II library allows much larger than simulations than typical finite element programs – PRISMS-PF has been used for simulations with over one billion degrees of freedom. PRISMS-PF also provides performance competitive with or exceeding single-purpose codes. For example, even without enabling the mesh adaptivity features in PRISMS-PF, it has been demonstrated to be 4.5x faster than an equivalent finite difference code.

  This code is developed by the PRedictive Integrated Structural
  Materials Science (PRISMS) Center [http://www.prisms-center.org/]
  at University of Michigan which is supported by the U.S. Department
  of Energy (DOE), Office of Basic Energy Sciences, Division of Materials Sciences
  and Engineering under Award #DE-SC0008637

<B>Quick Start Guide:</B>

For detailed instructions on how to download and use PRISMS-PF, please consult the PRISMS-PF User Guide (the file prismspf_user_guide.pdf). A (very) abbreviated version of the instructions is given below.

<I>Installation:</I>

1) Install deal.II (version 8.4.2 recommended)<br>
  + Download CMake [http://www.cmake.org/download/]
  + Add CMake to your path (e.g. $ PATH="/path/to/cmake/Contents/bin":"$PATH"), preferably in a shell configuration file
  + Download and install Deal.II following instructions from from https://www.dealii.org/download.html
  + If a Deal.II binary is downloaded, open it and follow the instructions in the terminal window
  + If Deal.II is installed from the source, the MPI and p4est libraries must be installed as prerequisites.
<br>

2) Clone the PRISMS-PF GitHub repo https://github.com/prisms-center/phaseField<br>
  + $ git clone https://github.com/prisms-center/phaseField.git <br>
  + $ cd phaseField <br>
  + $ git checkout master <br>

<I>Updates:</I>

Since PRISMS-PF is still under active development,
  regular code and documentation updates are pushed to the upstream
  repo (https://github.com/prisms-center/phaseField) and we strongly
  recommend users to synchronize their respective clones/forks at regular
  intervals or when requested by the developers through the
  announcements on the mailing list.

<I>Running a Pre-Built Application:</I>

  Entering the following commands will run one of the pre-built example applications (the Cahn-Hilliard spinodal decomposition application in this case):<br>
  + $ cd applications/cahnHilliard <br>
  + $ cmake CMakeLists.txt <br><br>
  For debug mode [default mode, very slow]: <br>
  + $ make debug <br><br>
  For optimized mode:<br>
   + $ make release <br><br>
  Execution (serial runs): <br>
  + $ ./main <br><br>
  Execution (parallel runs): <br>
  + $ mpirun -np nprocs main <br>
  [here nprocs denotes the number of processors]

<I>Visualization:</I>

  Output of the primal fields fields is in standard vtk
  format (parallel:*.pvtu, serial:*.vtu files) which can be visualized with the
  following open source applications:
  1. VisIt (https://wci.llnl.gov/simulation/computer-codes/visit/downloads)
  2. Paraview (http://www.paraview.org/download/)

<I>Getting started:</I>

  Examples of various phase field models are located under the
  applications/ folder. The easiest way to get started on the code is to
  run the example applications in this folder.

  THe example applications are intended to serve as (1) Demonstration of the
  capabilities of this library, (2) Provide a framework for
  further development of specialized/advanced applications by
  users.

  Applications that are still under development/testing are preceded by an
  underscore.

<B>Documentation:</B>

  The PRISMS-PF Users Guide provides extensive documentation on installing PRISMS-PF, running and visualizing simulations, and the structure of the input files.

  Doxygen-generated documentation can be viewed in one of two ways:
  + Open html/index.html in any web browser <br>
  (OR)<br>
  + https://htmlpreview.github.io/?https://raw.githubusercontent.com/prisms-center/phaseField/master/html/index.html

<B>License:</B>

  GNU Lesser General Public License (LGPL). Please see the file
  LICENSE for details.

<B>Further information, questions, issues and bugs:</B>

 + prisms-pf-users@googlegroups.com (user forum)
 + prismsphaseField.dev@umich.edu  (developer email list)
