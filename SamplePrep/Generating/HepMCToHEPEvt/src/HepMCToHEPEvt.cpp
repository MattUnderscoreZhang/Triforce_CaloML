#include "HepMCToHEPEvt.hh"

#include <fstream>


int main (int argc, char **argv) {

  if(argc < 3) {
    std::cout << "HepMCToHEPEvt <HepMC.dat> <events.hepevt>"  << std::endl;
    return 1;
  }
  std::ifstream InputFile(argv[1],  std::ios::in);
  std::ofstream outputFile(argv[2], std::ios::trunc);

  HepMC::GenEvent hepmcEvent;

  while(InputFile >> hepmcEvent) {

    HepMCToHEPEvtStream(hepmcEvent, outputFile);

  }//for all events

  return 0;
}
