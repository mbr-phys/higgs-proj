/*
 * lam-scan.cpp
 *
 * For an selected point of R2HDM parameter space in the mass basis (mH0,mh0,mA0,mH+,tan(beta),alpha)
 * find which values of lambdas 1->5 and m12 approximately correspond to this in the lambda basis
 * by calculating the mass terms; beta and alpha are prefixed by input and the alignment limit
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
    const string precise = "precision.dat";
    const string best_fit = "best_fit.dat";
//    const string out_name = "results.out";
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
    const double align = b - M_PI/2.0; //alignment limit
    double a,test;

    const int steps = 10; //steps for the lambdas
    const int steps2 = 400; //steps for m12
    double l1[steps],l2[steps],l3[steps],l4[steps],l5[steps],m12[steps2];

    //initialising each value of the arrays -- there must be a better way to do this?
    for (int i = 0; i < steps; i++)     
    {
        const double st = 8.0*M_PI/steps;
        l1[i] = -4.0*M_PI+st*i;
        l2[i] = -4.0*M_PI+st*i;
        l3[i] = -4.0*M_PI+st*i;
        l4[i] = -4.0*M_PI+st*i;
        l5[i] = -4.0*M_PI+st*i;
    }

    for (int j = 0; j < steps2; j++)
    {
        const double mt = 10000.0/steps2;
        m12[j] = mt*j;
    }

    //open output files
//    ofstream lamfile;
//    lamfile.open(lam_out.c_str());//,ios::app);

//    ofstream prefile;
//    prefile.open(precise.c_str());//,ios::app);

    ofstream bffile;
    bffile.open(best_fit.c_str(),ios::app);

    //define use angle values
    double t2a;
    double s2ab;// = pow(sin(a-b),2);
    double c2ab;// = pow(cos(a-b),2);
    double sb = sin(b);
    double cb = cos(b);
    double sa;// = sin(a);
    double ca;// = cos(a);
    double s2b = sin(2*b);
    double s2a;// = sin(2*a);

    //values to be calculated by lambda basis
    double M2,l345;
    double maH02,mah2,maA2,maHp2;
    double m112,m222;

    //booleans for theoretical constrains on lambdas
    bool b1,b2,b3,b4;
    //booleans for where each mass calculated is near mass read from file and alpha near align
    bool ho,hs,ao,hp,ab;

    int counting = 0;               //count overall iterations
    int counter = 0;                //counter of how many successful tests there are
    int lin = 0;                    //register which line of lambdas.dat has best fitting parameters
    double dho,dhs,dao,dhp,avg,daa;     //difference between calculated and input masses
    double dho2 = 200;              //holds calculated mass values of best fitting point
    double dhs2 = 200;
    double dao2 = 200;
    double dhp2 = 200;
    double avg2 = 200;              //average difference between calc and input masses
    double daa2 = 200;

    double l1b,l2b,l3b,l4b,l5b,m12b;    //best fitting parameter points
    double m112b,m222b;

    for (int i = 0; i < steps; i++) //lam1
    {
        for (int j = 0; j < steps; j++) //lam2
        {
            for (int c = 0; c < steps; c++) //lam3
            {
                for (int d = 0; d < steps; d++) //lam4
                {
                    for (int e = 0; e < steps; e++) //lam5
                    {
                        for (int f = 0; f < steps2; f++) //m12
                        {
                            counting++;
                            if (counting == pow(steps,5)*steps2/2.0) {
                                cout << "Don't worry, I'm still working on it. You're halfway through!" << endl;}

                            b1 = l1[i] > 0;
                            b2 = l2[j] > 0;
                            b3 = l3[c] > -1.0*sqrt(l1[i]*l2[j]);
                            b4 = (l3[c] + l4[d] - l5[e]) > -1.0*sqrt(l1[i]*l2[j]);
                            if (b1 && b2 && b3 && b4) {     //perturbativity constraints
                                M2 = m12[f]*m12[f]/(sb*cb);
                                l345 = (l3[c]+l4[d]+l5[e])/2.0;

                                t2a = 2.0*(-1*pow(m12[f],2) + 2.0*l345*vev*vev*cb*sb)/(pow(m12[f],2)*(tanb - 1.0/tanb) + l1[i]*pow(vev*cb,2) - l2[j]*pow(vev*sb,2));
                                a = 0.5*atan(t2a);
                                s2ab = pow(sin(a-b),2);
                                c2ab = pow(cos(a-b),2);
                                sa = sin(a);
                                ca = cos(a);
                                s2a = sin(2*a);

                                maH02 = M2*s2ab + pow(vev,2)*(l1[i]*pow(ca*cb,2) + l2[j]*pow(sa*sb,2) + l345*s2a*s2b);
                                mah2 = M2*c2ab + pow(vev,2)*(l1[i]*pow(sa*cb,2) + l2[j]*pow(ca*sb,2) - l345*s2a*s2b);
                                maA2 = M2 - l5[e]*pow(vev,2);
                                maHp2 = M2 - (l4[d]+l5[e])*pow(vev,2)/2.0;

                                m112 = pow(m12[f],2)*tanb - 0.5*pow(vev*cb*l1[i],2) - l345*pow(vev*sb,2);
                                m222 = pow(m12[f],2)*1.0/tanb - 0.5*pow(vev*sb*l2[j],2) - l345*pow(vev*cb,2);

                                //make sure no negative masses
                                if (maH02 < 0) {
                                    ho = false;}
                                else {
                                    ho = (mH0-100)<sqrt(maH02) && sqrt(maH02)<(mH0+100);}
                                if (mah2 < 0) {
                                    hs = false;}
                                else {
                                    hs = (hSM-100)<sqrt(mah2) && sqrt(mah2)<(hSM+100);}
                                if (maA2 < 0) {
                                    ao = false;}
                                else {
                                    ao = (mA0-100)<sqrt(maA2) && sqrt(maA2)<(mA0+100);}
                                if (maHp2 < 0) {
                                    hp = false;}
                                else {
                                    hp = (mHp-100)<sqrt(maHp2) && sqrt(maHp2)<(mHp+100);}

                                test = cos(b-a);
                                ab = fabs(test)<0.05; //Oliver's constraints

//                                if (ho && hs && ao && hp) {
//                                    cout << ho << " " << hs << " " << ao << " " << hp << " " << a/align << endl;}
                                if (ho && hs && ao && hp && ab)   //if all masses are close to input ones and alpha in alignment
                                {
                                    //output data to file for reading to BSMPT
//                                    lamfile << 2 << " " << l1[i] << " " << l2[j] << " " << l3[c] << " " << l4[d] << " " 
//                                            << l5[e] << " " << pow(m12[f],2) << " " << tanb << " " << sqrt(maH02) << " " 
//                                            << sqrt(mah2) << " " << sqrt(maA2) << " " << sqrt(maHp2) << " " << a << " " 
//                                            << m112 << " " << m222  << " "
//                                            << bs_gamma << " " << maxev << " " << b_h_ss << " " << b_h_cc << " " << b_h_bb << " " 
//                                            << b_h_mumu << " " << b_h_tautau << " " << b_h_WW << " " << b_h_ZZ << " " 
//                                            << b_h_Zga << " " << b_h_gaga << " " << b_h_gg << " " << b_h_AA << " " << w_h << " " 
//                                            << b_A_ss << " " << b_A_cc << " " << b_A_bb << " " << b_A_tt << " " << b_A_mumu << " " 
//                                            << b_A_tautau << " " << b_A_WW << " " << b_A_ZZ << " " << b_A_Zga << " " 
//                                            << b_A_gaga << " " << b_A_gg << " " << w_A << " " << b_H_ss << " " << b_H_cc << " " 
//                                            << b_H_bb << " " << b_H_tt << " " << b_H_mumu << " " << b_H_tautau << " " 
//                                            << b_H_WW << " " << b_H_ZZ << " " << b_H_Zga << " " << b_H_gaga << " " << b_H_gg << " "
//                                            << b_H_hh << " " << b_H_AA << " " << w_H << " " << b_t_Wb << " " << b_t_Hcb << " " 
//                                            << b_Hc_cs << " " << b_Hc_cb << " " << b_Hc_tb << " " << b_Hc_ts << " " 
//                                            << b_Hc_tau << " " << b_Hc_Wh << " " << b_Hc_WH << " " << b_Hc_WA << " " << w_Hc << " " 
//                                            << mu_F << " " << mu_V << " " << mu_gaga << " " << mu_ZZ << " " << mu_WW << " " 
//                                            << mu_tautau << " " << mu_bb << " " << x_h_ggF << " " << x_A_ggF << " " 
//                                            << x_H_ggF << " " << endl;
                                    counter++;                      //another successful test
                                    dho = fabs(sqrt(maH02)-mH0);    //differences between calc and input masses
                                    dhs = fabs(sqrt(mah2)-hSM);
                                    dao = fabs(sqrt(maA2)-mA0);
                                    dhp = fabs(sqrt(maHp2)-mHp);
                                    daa = fabs(a-align);

                                    //output to precision file in case we're interested
//                                    prefile << "Line Number: " << counter 
//                                            << "; D(mH0): " << dho
//                                            << "; D(mh0): " << dhs
//                                            << "; D(mA0): " << dao
//                                            << "; D(mH+): " << dhp 
//                                            << "; D(alp): " << daa << endl;
                                    avg = (dho+dhs+dao+dhp+daa)/5.0;    //average best fitting point 
//                                    avg = dhs;                      //closest to getting SM Higgs right

                                    //set information for best fitting point
                                    if (avg < avg2) {
                                        avg2 = avg;
                                        l1b = l1[i];
                                        l2b = l2[j];
                                        l3b = l3[c];
                                        l4b = l4[d];
                                        l5b = l5[e];
                                        m12b = m12[f];
                                        m112b = m112;
                                        m222b = m222;
                                        dho2 = sqrt(maH02);
                                        dhs2 = sqrt(mah2);
                                        dao2 = sqrt(maA2);
                                        dhp2 = sqrt(maHp2);
                                        daa2 = a;
                                        lin = counter;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    double theta = (b-daa2)*180.0/M_PI;
    cout << counter << endl;
    cout << "Closest fitting point is at line " << lin << endl
         << "mH0: " << dho2 << "; mh0: " << dhs2 << "; mA0: " << dao2 << "; mH+:" << dhp2 << "; theta: " << theta << endl;

    bffile << 2 << " " << l1b << " " << l2b << " " << l3b << " " << l4b << " " << l5b << " " << pow(m12b,2) << " " 
        << tanb << " " << dho2 << " " << dhs2 << " " << dao2 << " " << dhp2 << " " << daa2 << " " << m112b << " " 
        << m222b  << " " << bs_gamma << " " << maxev << " " << b_h_ss << " " << b_h_cc << " " << b_h_bb << " " << b_h_mumu << " "
        << b_h_tautau << " " << b_h_WW << " " << b_h_ZZ << " " << b_h_Zga << " " << b_h_gaga << " " << b_h_gg << " " 
        << b_h_AA << " " << w_h << " " << b_A_ss << " " << b_A_cc << " " << b_A_bb << " " << b_A_tt << " " << b_A_mumu << " " 
        << b_A_tautau << " " << b_A_WW << " " << b_A_ZZ << " " << b_A_Zga << " " << b_A_gaga << " " << b_A_gg << " " 
        << w_A << " " << b_H_ss << " " << b_H_cc << " " << b_H_bb << " " << b_H_tt << " " << b_H_mumu << " " << b_H_tautau << " "
        << b_H_WW << " " << b_H_ZZ << " " << b_H_Zga << " " << b_H_gaga << " " << b_H_gg << " " << b_H_hh << " " << b_H_AA << " "
        << w_H << " " << b_t_Wb << " " << b_t_Hcb << " " << b_Hc_cs << " " << b_Hc_cb << " " << b_Hc_tb << " " << b_Hc_ts << " " 
        << b_Hc_tau << " " << b_Hc_Wh << " " << b_Hc_WH << " " << b_Hc_WA << " " << w_Hc << " " << mu_F << " " << mu_V << " " 
        << mu_gaga << " " << mu_ZZ << " " << mu_WW << " " << mu_tautau << " " << mu_bb << " " << x_h_ggF << " " << x_A_ggF << " "
        << x_H_ggF << " " << endl;

//    lamfile.close();
//    prefile.close();
    bffile.close();
    //run BSMPT -- change file names
//    system("./../../../BSMPT/build/bin/BSMPT r2hdm ~/library/higgs-proj/EWPT/lambdas.dat ~/library/higgs-proj/EWPT/scanPT.dat 1 -1");
}
