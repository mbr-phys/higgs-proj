/*
 *  Outdated first attempt of getting lambda basis for 2HDM based on
 *  mass basis parameter points
 *  Use lam-scan.cpp for more general results
 *
 */

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <string>

using namespace std;

int main()
{
    const string in_name = "mass_basis.in";
    const string out_name = "lam_basis.out";
    const double hSM = 125.1;
    const double vev = 246;

    //set up top line of output file
    ofstream outfile;
    outfile.open(out_name.c_str(),ios::app);
//    outfile << "type " << "L1 " << "L2 " << "L3 " << "L4 " << "L5 " << "m12sq " << "tbeta " 
//            << "mH " << "mh " << "mA " << "mHc " << "alpha " << endl;
    
    cout << "Reading in data" << endl;
    ifstream infile;
    infile.open(in_name.c_str());

    while (true)
    {
        double mHp,mH0,mA0,tanb,beta;
        infile >> mHp >> mH0 >> mA0 >> tanb;

        if (infile.eof()) break;

        const double l1 = 0;
        const double l2 = 0;
        double l3,l4,l5,m12;
        beta = atan(tanb);

        m12 = sqrt((mH0*mH0 + hSM*hSM)*sin(beta)*cos(beta));
        l5 = (mH0*mH0 - mA0*mA0 + hSM*hSM)/(vev*vev);
        l4 = (mH0*mH0 + mA0*mA0 - 2*mHp*mHp + hSM*hSM)/(vev*vev);

        double alpha = beta - M_PI/2;
        double pref = sin(2*alpha)*sin(2*beta)*vev*vev;
        l3 = (m12*m12*(pow(sin(alpha-beta),2) - pow(cos(alpha-beta),2))/(cos(beta)*sin(beta)) + hSM*hSM - mH0*mH0)/pref - l4 - l5;

        if (-sqrt(l1*l2) > l3 || abs(l5)-l4-sqrt(l1*l2) > l3)
        {
            cout << "lamba3 violating perturbativity" << endl;
        }

        outfile << 2 << " " << 0 << " " << 0 << " " << l3 << " " << l4 << " " << l5 << " " << m12*m12 
                << " " << tanb << " " << mH0 << " " << hSM << " " << mA0 << " " << mHp << " " << alpha << endl;
    }

    //execute BSMPT?

    infile.close();
    outfile.close();
}
