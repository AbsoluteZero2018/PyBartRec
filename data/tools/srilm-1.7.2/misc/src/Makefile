#
#    File:   Makefile.example
#    Author: The SRI DECIPHER (TM) System
#    Date:   Thu Sep  9 12:04:47 1993
#
#    Description:
#	This is the example makefile to start from when adding new
#       modules to the DECIPHER System.  To use this makefile, first
#       copy it to your directory as the file "Makefile".  Second,
#       replace the word "Example" in the text below with the real name
#	of your library.  Next replace the the example filenames with
#	the names of your actual declarations and source files in the
#	appropriate variable definitions.  Finally clean up by deleting
#	any lines not relevant to your module and updating this header
#       to describe your new module.  Do not forget to use the proper
#       RCS keywords!
#
#    Copyright (c) 1993, SRI International.  All Rights Reserved.
#
#    $Header: /home/srilm/CVS/srilm/misc/src/Makefile,v 1.38 2016/05/25 16:26:15 stolcke Exp $
#

# Include common SRILM variable definitions.
include $(SRILM)/common/Makefile.common.variables

# Define variables.

# Example Library declarations files.

EXTERNAL_LIB_HEADERS = \
	$(SRCDIR)/cfuncproto.h          \
	$(SRCDIR)/option.h		\
	$(SRCDIR)/zio.h			\
	$(SRCDIR)/mkdir.h		\
	$(SRCDIR)/Boolean.h		\
	$(SRCDIR)/Debug.h		\
	$(SRCDIR)/File.h		\
	$(SRCDIR)/MStringTokUtil.h	\
	$(SRCDIR)/tls.h                 \
	$(SRCDIR)/TLSWrapper.h          \
	$(SRCDIR)/tserror.h             \
	$(SRCDIR)/version.h

INTERNAL_LIB_HEADERS =  \
	$(SRCDIR)/srilm_iconv.h	

# Example Library source files.
LIB_SOURCES = 				\
	$(SRCDIR)/option.c		\
	$(SRCDIR)/zio.c			\
	$(SRCDIR)/fcheck.c		\
	$(SRCDIR)/rand48.c		\
	$(SRCDIR)/Debug.cc		\
	$(SRCDIR)/File.cc		\
	$(SRCDIR)/MStringTokUtil.cc	\
	$(SRCDIR)/tls.cc        	\
	$(SRCDIR)/tserror.cc        	\
	$(SRCDIR)/version.c

ifeq ($(NO_TCL), )
LIB_SOURCES +=	$(SRCDIR)/tclmain.cc
endif

ifneq ($(NO_ICONV), )
  CFLAGS += -DNO_ICONV
  CXXFLAGS += -DNO_ICONV
endif

# tell version.c about compile-time options
ifneq ($(HAVE_LIBLBFGS), )
  ADDITIONAL_CFLAGS += -DHAVE_LIBLBFGS
  ADDITIONAL_CXXFLAGS += -DHAVE_LIBLBFGS
endif

# Example Library object files.
LIB_OBJECTS = $(patsubst $(SRCDIR)/%.cc,$(OBJDIR)/%$(OBJ_SUFFIX),\
		$(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%$(OBJ_SUFFIX),$(LIB_SOURCES)))

# Example Library.
LIBRARY = $(OBJDIR)/$(LIB_PREFIX)misc$(LIB_SUFFIX)

ADDITIONAL_INCLUDES += \
	$(TCL_INCLUDE)

ADDITIONAL_LIBRARIES += \
	$(SRILM_LIBDIR)/$(LIB_PREFIX)z$(LIB_SUFFIX) \
	$(TCL_LIBRARY) \
	$(MATH_LIBRARY)

# Example programs.
PROGRAM_NAMES = \
	ztest \
	testFile \
	testRand

PROGRAMS = $(PROGRAM_NAMES:%=$(BINDIR)/%$(EXE_SUFFIX))

PROGRAM_SOURCES = $(foreach prog,$(PROGRAM_NAMES),\
			$(wildcard $(SRCDIR)/$(prog).c) \
			$(wildcard $(SRCDIR)/$(prog).cc))
PROGRAM_OBJECTS = $(PROGRAM_NAMES:%=$(OBJDIR)/%$(OBJ_SUFFIX))


# Libraries to be linked with the Example programs.
LIBRARIES = $(LIBRARY) \
	    $(ADDITIONAL_LIBRARIES)

# All of the types of files.

ALL_HEADERS = $(EXTERNAL_LIB_HEADERS) \
	      $(INTERNAL_LIB_HEADERS)

ALL_SOURCES = $(LIB_SOURCES) \
	      $(PROGRAM_SOURCES)

ALL_OBJECTS = $(LIB_OBJECTS) \
	      $(PROGRAM_OBJECTS)

ALL_LIBRARIES = $(LIBRARY)

ALL_PROGRAMS = $(PROGRAMS)

ALL_PROGRAM_NAMES = $(PROGRAM_NAMES)


# Define pseudo-targets.


# Make sure the library does not get deleted if the make is interrupted.
.PRECIOUS:	$(LIBRARY)


# Define targets.


all:	libraries

$(LIBRARY):	$(LIB_OBJECTS)
	$(ARCHIVE) $(AR_OUTPUT_OPTION) $^
	$(RANLIB) $@ $(DEMANGLE_FILTER)

SRILMversion.h:	$(SRILM)/RELEASE $(SRILM)/Copyright
	read version < $(SRILM)/RELEASE; echo "#define SRILM_RELEASE \"$$version\"" > $@
	sed -f $(SRILM)/sbin/stringify-copyright $(SRILM)/Copyright >> $@

# Variables and Targets for released system

EXPORTED_HEADERS   = $(EXTERNAL_LIB_HEADERS)
EXPORTED_LIBRARIES = $(LIBRARY)
EXPORTED_PROGRAMS  = 

release:	release-headers release-libraries release-programs

depend:	SRILMversion.h

cleaner:
	rm -f SRILMversion.h

# Include common SRILM target definitions.
include $(SRILM)/common/Makefile.common.targets

