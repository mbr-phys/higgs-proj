/*
 * lam-conv.cpp
 *
 * Convert mass basis of 2HDM II to lambda basis, using eqns in hep-ph/2003.06170 and vacuum stability
 * 
 * Outputs these to a file, formatted to be read by the BSMPT program (https://github.com/phbasler/BSMPT)
 * Also outputs a file containing the deviation for each mass calculated from the input value
 */

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

using namespace std;

int main()
{
    //filenames and basic constants
    const string in_name = "mass_basis.in";
    const string r2h = "R2HDM_Input.dat";
    const string lam_out = "lambdas.dat";
    const double hSM = 125.1;
    const double vev = 246.21965079413735;

    //reading from BSMPT example for extra values -- update values to more recent?
    double type,La1,La2,La3,La4,La5,m12sq,tbeta,mH,mhs,mA,mHc,alp,m11sq,m22sq;
    double bs_gamma,maxev,b_h_ss,b_h_cc,b_h_bb,b_h_mumu,b_h_tautau,b_h_WW,b_h_ZZ,b_h_Zga,b_h_gaga,b_h_gg,b_h_AA,w_h,b_A_ss,b_A_cc,b_A_bb,b_A_tt,b_A_mumu,b_A_tautau,b_A_WW,b_A_ZZ,b_A_Zga,b_A_gaga,b_A_gg,w_A,b_H_ss,b_H_cc,b_H_bb,b_H_tt,b_H_mumu,b_H_tautau,b_H_WW,b_H_ZZ,b_H_Zga,b_H_gaga,b_H_gg,b_H_hh,b_H_AA,w_H,b_t_Wb,b_t_Hcb,b_Hc_cs,b_Hc_cb,b_Hc_tb,b_Hc_ts,b_Hc_tau,b_Hc_Wh,b_Hc_WH,b_Hc_WA,w_Hc,mu_F,mu_V,mu_gaga,mu_ZZ,mu_WW,mu_tautau,mu_bb,x_h_ggF,x_A_ggF,x_H_ggF;
    ifstream input_file;
    input_file.open(r2h.c_str());
    input_file >> type >> La1 >> La2 >> La3 >> La4 >> La5 >> m12sq >> tbeta >> mH >> mhs >> mA >> mHc >> alp >> m11sq >> m22sq
               >> bs_gamma >> maxev >> b_h_ss >> b_h_cc >> b_h_bb >> b_h_mumu >> b_h_tautau >> b_h_WW >> b_h_ZZ >> b_h_Zga
               >> b_h_gaga >> b_h_gg >> b_h_AA >>w_h >> b_A_ss >> b_A_cc >> b_A_bb >> b_A_tt >> b_A_mumu >> b_A_tautau 
               >> b_A_WW >> b_A_ZZ >> b_A_Zga >> b_A_gaga >> b_A_gg >> w_A >> b_H_ss >> b_H_cc >> b_H_bb >> b_H_tt >> b_H_mumu 
               >> b_H_tautau >> b_H_WW >> b_H_ZZ >> b_H_Zga >> b_H_gaga >> b_H_gg >> b_H_hh >> b_H_AA >> w_H >> b_t_Wb 
               >> b_t_Hcb >> b_Hc_cs >> b_Hc_cb >> b_Hc_tb >> b_Hc_ts >> b_Hc_tau >> b_Hc_Wh >> b_Hc_WH >> b_Hc_WA >> w_Hc
               >> mu_F >> mu_V >> mu_gaga >> mu_ZZ >> mu_WW >> mu_tautau >> mu_bb >> x_h_ggF >> x_A_ggF >> x_H_ggF;
    input_file.close();
    
    //reading in mass basis parameter point -- only reads in one right now
    double mHp,mH0,mA0,tanb;
    ifstream infile;
    infile.open(in_name.c_str());
    infile >> mHp >> mH0 >> mA0 >> tanb;
    infile.close();
//    cout << "Input mH+, mH0, mA0, beta: " << endl;
//    cin >> mHp >> mH0 >> mA0 >> b;
    const double b = atan(tanb);
    double alpha = b - M_PI/2.0;

    //define use angle values
    double sb = sin(b);
    double cb = cos(b);

    double m12 = mH0*mH0*sb*cb;
    double l1 = hSM*hSM/(vev*vev);
    double l2 = hSM*hSM/(vev*vev);
    double l3 = (hSM*hSM + 2*mHp*mHp - 2*mH0*mH0)/(vev*vev);
    double l4 = (mA0*mA0 - 2*mHp*mHp + mH0*mH0)/(vev*vev);
    double l5 = (mH0*mH0 - mA0*mA0)/(vev*vev);

    //open output file
    ofstream bffile;
    bffile.open(lam_out.c_str(),ios::app);

    double m112 = m12*tanb - 0.5*l1*l1*vev*vev*cb*cb - 0.5*vev*vev*sb*sb*(l3+l4+l5);
    double m222 = m12*1.0/tanb - 0.5*l2*l2*vev*vev*sb*sb - 0.5*vev*vev*cb*cb*(l3+l4+l5);

    bffile << 2 << " " << l1 << " " << l2 << " " << l3 << " " << l4 << " " << l5 << " " << m12 << " " 
        << tanb << " " << mH0 << " " << hSM << " " << mA0 << " " << mHp << " " << alpha << endl; 
//    " " << m112 << " " 
//        << m222  << " " << bs_gamma << " " << maxev << " " << b_h_ss << " " << b_h_cc << " " << b_h_bb << " " << b_h_mumu << " "
//        << b_h_tautau << " " << b_h_WW << " " << b_h_ZZ << " " << b_h_Zga << " " << b_h_gaga << " " << b_h_gg << " " 
//        << b_h_AA << " " << w_h << " " << b_A_ss << " " << b_A_cc << " " << b_A_bb << " " << b_A_tt << " " << b_A_mumu << " " 
//        << b_A_tautau << " " << b_A_WW << " " << b_A_ZZ << " " << b_A_Zga << " " << b_A_gaga << " " << b_A_gg << " " 
//        << w_A << " " << b_H_ss << " " << b_H_cc << " " << b_H_bb << " " << b_H_tt << " " << b_H_mumu << " " << b_H_tautau << " "
//        << b_H_WW << " " << b_H_ZZ << " " << b_H_Zga << " " << b_H_gaga << " " << b_H_gg << " " << b_H_hh << " " << b_H_AA << " "
//        << w_H << " " << b_t_Wb << " " << b_t_Hcb << " " << b_Hc_cs << " " << b_Hc_cb << " " << b_Hc_tb << " " << b_Hc_ts << " " 
//        << b_Hc_tau << " " << b_Hc_Wh << " " << b_Hc_WH << " " << b_Hc_WA << " " << w_Hc << " " << mu_F << " " << mu_V << " " 
//        << mu_gaga << " " << mu_ZZ << " " << mu_WW << " " << mu_tautau << " " << mu_bb << " " << x_h_ggF << " " << x_A_ggF << " "
//        << x_H_ggF << endl;

//    lamfile.close();
//    prefile.close();
    bffile.close();
    //run BSMPT -- change file names
//    system("./../../../BSMPT/build/bin/BSMPT r2hdm ~/library/higgs-proj/EWPT/lambdas.dat ~/library/higgs-proj/EWPT/scanPT.dat 1 -1");
}
