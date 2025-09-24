# get hostname
HOST=$(shell hostname -f)

# Gadi
ifneq (,$(findstring gadi,$(HOST)))
CXX       = mpicxx
HDF5_PATH = $(HDF5_BASE)
SZIP_PATH = $(SZIP_BASE)
CCFLAGS = -O3
LDFLAGS = -O3
CCFLAGS += -I$(HDF5_PATH)/include -I$(SZIP_PATH)
LDFLAGS += -L$(HDF5_PATH)/lib -lhdf5 -lz -L$(SZIP_PATH) -lsz
endif

# Setonix
ifneq (,$(findstring setonix,$(HOST)))
Setonix = yes
endif
ifneq (,$(findstring dm,$(HOST)))
Setonix = yes
endif
ifneq (,$(findstring nid,$(HOST)))
Setonix = yes
endif
ifdef Setonix
CXX = CC
HDF5_PATH = $(HDF5_DIR)
CCFLAGS = -O3
LDFLAGS = -O3
CCFLAGS += -I$(HDF5_PATH)/include
LDFLAGS += -L$(HDF5_PATH)/lib -lhdf5 -lz
endif

# SuperMUC
ifneq (,$(findstring sng.lrz.de,$(HOST)))
CXX       = mpiCC
HDF5_PATH = $(HDF5_BASE)
SZIP_PATH = $(SZIP_BASE)
CCFLAGS = -O3
LDFLAGS = -O3
CCFLAGS += -I$(HDF5_PATH)/include
LDFLAGS += -L$(HDF5_PATH)/lib -lhdf5 -lz
endif

# MSO
ifneq (,$(findstring .anu.edu.au,$(HOST)))
mso_or_ozstar = yes
endif
# OzStar
ifneq (,$(findstring farnarkle,$(HOST)))
mso_or_ozstar = yes
endif
# MSO or OzStar
ifdef mso_or_ozstar
CXX = mpicxx
CCFLAGS = -O3 -g -Wall
LDFLAGS = -O3 -g -Wall
CCFLAGS += 
LDFLAGS += -lhdf5 -lz -lsz
endif

# Twister
ifneq (,$(findstring Twister,$(HOST)))
Twister = yes
endif
ifneq (,$(findstring minitron,$(HOST)))
Twister = yes
endif
ifdef Twister
CXX = mpicxx
HDF5_PATH = /opt/local
SZIP_PATH = /opt/local
CCFLAGS = -O3 -g -Wall
LDFLAGS = -O3 -g -Wall
CCFLAGS += -I$(HDF5_PATH)/include -I$(SZIP_PATH)
LDFLAGS += -L$(HDF5_PATH)/lib -lhdf5 -lz -L$(SZIP_PATH) -lsz
endif

BINS = downsample

all : $(BINS)

clean :
	rm -rf *.o *.dSYM *~ $(BINS)

$(BINS): %: %.cpp
	$(CXX) -o $@ $< $(CCFLAGS) $(LDFLAGS)
