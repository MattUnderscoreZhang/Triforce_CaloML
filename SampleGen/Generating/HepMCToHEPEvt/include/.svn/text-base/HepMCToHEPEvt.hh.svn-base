#ifndef HepMCToHEPEvt_hh
#define HepMCToHEPEvt_hh 1

#include <HepMC/IO_GenEvent.h>
#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>
#include <HepMC/GenVertex.h>


#include <iostream>
#include <iomanip>


///Writes the hepmcEvent to the output stream
/// Thans to L.Weuste for most of the code
///http://forum.linearcollider.org/index.php?t=msg&th=706&goto=1955&rid=0#msg_1955
void HepMCToHEPEvtStream(HepMC::GenEvent const& hepmcEvent, std::ostream& output) {

  int nParticles = hepmcEvent.particles_size();
  int nVertices =  hepmcEvent.vertices_size();
  int eventNumber = hepmcEvent.event_number();

  std::cout << std::setw(10) << eventNumber
	    << std::setw(10) << nParticles
	    << std::setw(10) << nVertices
	    << std::endl;

  output << nParticles << std::endl;      

  for (HepMC::GenEvent::particle_const_iterator particles = hepmcEvent.particles_begin(); 
       particles != hepmcEvent.particles_end() ; ++particles) {
    HepMC::GenParticle& particle = **particles;

    const int status = particle.status();
    HepMC::GenVertex* vertexProd = particle.production_vertex();
    HepMC::GenVertex* vertexDecay = particle.end_vertex();

    output << std::setw(4) << status << ' ';    // ISTHEP, status code
    output << std::setw(8) << particle.pdg_id() << ' '; // IDHEP, PDG code

    if(vertexProd) {
      output << std::setw(4) << (*(vertexProd->particles_in_const_begin()))->barcode()-10000 << ' '; // JMOHEP1, first mother id
      output << std::setw(4) << (*(vertexProd->particles_in_const_end()-1))->barcode()-10000 << ' '; // JMOHEP2, last mother id
    } else {
      output << std::setw(4) << 0 << ' ';   // JMOHEP1, first mother id
      output << std::setw(4) << 0 << ' ';   // JMOHEP2, last mother id
    }

    if(vertexDecay) {
      output << std::setw(4) << (*(vertexDecay->particles_out_const_begin()))->barcode()-10000 << ' '; // JMOHEP1, first mother id
      output << std::setw(4) << (*(vertexDecay->particles_out_const_end()-1))->barcode()-10000 << ' '; // JMOHEP2, last mother id
    } else {
      output << std::setw(4) << 0 << ' ';   // JMOHEP1, first mother id
      output << std::setw(4) << 0 << ' ';   // JMOHEP2, last mother id
    }
      
    // now start the info in double, so switch the output to that format
    output.setf(std::ios_base::scientific);
    output.precision(8);

    output << std::setw(15) << particle.momentum().px()  << ' ';         // PHEP1, px in GeV/c
    output << std::setw(15) << particle.momentum().py()  << ' ';         // PHEP2, py in GeV/c
    output << std::setw(15) << particle.momentum().pz()  << ' ';         // PHEP3, pz in GeV/c
    output << std::setw(15) << particle.momentum().e()   << ' ';          // PHEP4, energy in GeV
    output << std::setw(15) << particle.generated_mass() << ' ';          // PHEP5, mass in GeV/cc
    
    if (vertexProd) {
      output << std::setw(15) << vertexProd->position().x() << ' ';      // VHEP1, x vertex pos in mm
      output << std::setw(15) << vertexProd->position().y() << ' ';      // VHEP2, y vertex pos in mm
      output << std::setw(15) << vertexProd->position().z() << ' ';      // VHEP3, z vertex pos in mm
      output << std::setw(15) << vertexProd->position().t();      // VHEP4, production time in mm/c
    } else {
      if (vertexDecay->position().t() < 1e-60) {
	output << std::setw(15) << vertexDecay->position().x() << ' ';      // VHEP1, x vertex pos in mm
	output << std::setw(15) << vertexDecay->position().y() << ' ';      // VHEP2, y vertex pos in mm
	output << std::setw(15) << vertexDecay->position().z() << ' ';      // VHEP3, z vertex pos in mm
	output << std::setw(15) << vertexDecay->position().t();      // VHEP4, production time in mm/c
      } else { // if the particle had some lifetime we are going to assume it came from  0 0 0
	output << std::setw(15) << 0.0 << ' ';      // VHEP1, x vertex pos in mm
	output << std::setw(15) << 0.0 << ' ';      // VHEP2, y vertex pos in mm
	output << std::setw(15) << 0.0 << ' ';      // VHEP3, z vertex pos in mm
	output << std::setw(15) << vertexDecay->position().t();      // VHEP4, production time in mm/c
      }
    }

    output << "\n";
  }//for all particles

}//GenEventToHEPEVT



#endif // HepMCToHEPEvt_hh
