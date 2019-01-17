CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

PROJECT( HepMCToHEPEvt )

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")

SET( PROJECT_VERSION_MAJOR 0 )
SET( PROJECT_VERSION_MINOR 1 )

INCLUDE_DIRECTORIES ( ${PROJECT_SOURCE_DIR}/include )

IF( "${HepMC_DIR}" STREQUAL "")
  MESSAGE (FATAL_ERROR " Set HepMC_DIR variable with -DHepMC_DIR=<location>" )
ENDIF( "${HepMC_DIR}" STREQUAL "")

SET(HepMC_LIB_DIR "${HepMC_DIR}/lib")

INCLUDE_DIRECTORIES( SYSTEM ${HepMC_DIR}/include )

FIND_LIBRARY( HepMC_LIB2
	NAMES HepMC
	PATHS ${HepMC_DIR}
	PATH_SUFFIXES lib
	NO_DEFAULT_PATH
)

LINK_LIBRARIES( ${HepMC_LIB2} )

SET( CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} )
#LIST ( APPEND CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Wshadow -ansi -Wno-long-long -Wuninitialized -fopenmp -ftree-vectorizer-verbose=0 -mavx -fno-stack-protector" )
LIST ( APPEND CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Wshadow -ansi -Wno-long-long -Wuninitialized -flto -Weffc++ --std=c++11 -ggdb" )
LIST ( APPEND CMAKE_LD_FLAGS "-flto -O3" )

ADD_SUBDIRECTORY ( ${PROJECT_SOURCE_DIR}/src )

MESSAGE (STATUS "*** Build Type ${CMAKE_BUILD_TYPE} ***" )
MESSAGE (STATUS "*** Compiler Flags:  ${CMAKE_CXX_FLAGS}"  )

