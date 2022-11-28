//
// Created by Noah Kubli on 24.11.22.
//

#include "cooling.hpp"
#include "cooler.h"

namespace cooling {

    template<typename T>
    struct Cooler<T>::Impl {
        friend struct Cooler<T>;
    private:
        //! @brief Solar mass in g
        constexpr static T ms_g = 1.989e33;
        //! @brief kpc in cm
        constexpr static T kp_cm = 3.086e21;
        //! @brief Gravitational constant in cgs units
        constexpr static T G_newton = 6.674e-8;
        //! @brief code unit mass in solar masses
        T ms = 1e16;
        //! @brief code unit length in kpc
        T kpc = 46400.;
        //! @brief Path to Grackle data file
        std::string grackle_data_file_path =
                CMAKE_SOURCE_DIR "/extern/grackle/grackle_repo/input/CloudyData_UVB=HM2012.h5";

        void initOptions(const chemistry_data &grackle_options);

        void initOptions(const std::string &grackle_options_file_path);

        GlobalValues global_values;

        void init(const double ms_sim, const double kp_sim, const int comoving_coordinates,
                  const std::optional<chemistry_data_> grackleOptions = std::nullopt,
                  const std::optional<std::string> grackleOptionsFile = std::nullopt,
                  const std::optional<double> t_sim = std::nullopt);

        chemistry_data getDefaultChemistryData() {
            chemistry_data data_default = _set_default_chemistry_parameters();
            data_default.grackle_data_file = &grackle_data_file_path[0];
            return data_default;
        }

        GlobalValues &get_global_values() {
            return global_values;
        }
    };


    //Implementation of Cooler
    template<typename T>
    Cooler<T>::Cooler() : impl_ptr(new Impl) {}

    template<typename T>
    Cooler<T>::~Cooler() = default;

    template<typename T>
    void Cooler<T>::init(const double ms_sim, const double kp_sim, const int comoving_coordinates,
                         const std::optional<chemistry_data_> grackleOptions,
                         const std::optional<std::string> grackleOptionsFile,
                         const std::optional<double> t_sim) {
        impl_ptr->init(ms_sim, kp_sim, comoving_coordinates,
                       grackleOptions,
                       grackleOptionsFile, t_sim);

    }

    template<typename T>
    chemistry_data_ Cooler<T>::getDefaultChemistryData() {
        chemistry_data_ data = {impl_ptr->getDefaultChemistryData()};
        return data;
    }

    template<typename T>
    GlobalValues &Cooler<T>::get_global_values() {
        return impl_ptr->get_global_values();
    }

    template
    struct Cooler<double>;
    template
    struct Cooler<float>;


    //Implementation of Cooler::Impl
    template<typename T>
    void Cooler<T>::Impl::init(const double ms_sim, const double kp_sim, const int comoving_coordinates,
                               const std::optional<chemistry_data_> grackleOptions,
                               const std::optional<std::string> grackleOptionsFile, const std::optional<double> t_sim) {
        ms = ms_sim;
        kpc = kp_sim;
        grackle_verbose = 1;

        // Density
        const double density_unit = ms * ms_g / std::pow(kpc * kp_cm, 3);
        // Time
        const double time_unit = t_sim.value_or(std::sqrt(1. / (density_unit * G_newton)));
        // Length
        const double length_unit = kpc * kp_cm;
        // Velocity
        const double velocity_unit = length_unit / time_unit;

        global_values.units.density_units = density_unit; // m_sun / (pc * pc * pc);
        global_values.units.time_units = time_unit;    // code_time;
        global_values.units.length_units = length_unit;  // pc;
        global_values.units.velocity_units = velocity_unit;
        global_values.units.a_units = 1.0;
        global_values.units.a_value = 1.0;
        global_values.units.comoving_coordinates = comoving_coordinates;

        std::cout << "debug\n";
        std::cout << ms << "\t" << ms_g << "\t" << kpc << "\n";
        std::cout << "code units\n";
        std::cout << global_values.units.density_units << "\t" << global_values.units.time_units << "\t"
                  << global_values.units.length_units << "\n";

        global_values.data = _set_default_chemistry_parameters();

        global_values.data.grackle_data_file = &grackle_data_file_path[0];

        if (grackleOptions.has_value() && grackleOptionsFile.has_value())
            throw std::runtime_error("Specify only one; either grackleOptions or grackleOptionsFile");

        if (grackleOptions.has_value()) { initOptions(grackleOptions.value().content); }
        else if (grackleOptionsFile.has_value()) { initOptions(grackleOptionsFile.value()); }
        else { initOptions(getDefaultChemistryData()); }

        if (0 == _initialize_chemistry_data(&global_values.data, &global_values.rates, &global_values.units)) {
            std::cout << global_values.data.with_radiative_cooling << std::endl;
            throw std::runtime_error("Grackle: Error in _initialize_chemistry_data");
        }
    }

    template<typename T>
    void Cooler<T>::Impl::initOptions(const chemistry_data &grackle_options) {
        global_values.data = grackle_options;
        grackle_data_file_path = std::string(grackle_options.grackle_data_file);
        std::cout << grackle_data_file_path << std::endl;
        global_values.data.grackle_data_file = &grackle_data_file_path[0];
    }

    template<typename T>
    void Cooler<T>::Impl::initOptions(const std::string &grackle_options_file_path) {
        auto setGrackleOption = [&](chemistry_data &data, const std::string &key, const std::string &value) {
            if (key.find("grackle_data_file_path") != std::string::npos) {
                grackle_data_file_path = std::string(value);
                data.grackle_data_file = &grackle_data_file_path[0];
            }
            if (key.find("use_grackle") != std::string::npos) data.use_grackle = std::stoi(value);
            if (key.find("with_radiative_cooling") != std::string::npos) data.with_radiative_cooling = std::stoi(value);
            if (key.find("primordial_chemistry") != std::string::npos) data.primordial_chemistry = std::stoi(value);
            if (key.find("h2_on_dust") != std::string::npos) data.h2_on_dust = std::stoi(value);
            if (key.find("metal_cooling") != std::string::npos) data.metal_cooling = std::stoi(value);
            if (key.find("cmb_temperature_floor") != std::string::npos) data.cmb_temperature_floor = std::stoi(value);
            if (key.find("UVbackground") != std::string::npos) data.UVbackground = std::stoi(value);
            if (key.find("UVbackground_redshift_on") != std::string::npos)
                data.UVbackground_redshift_on = std::stoi(value);
            if (key.find("UVbackground_redshift_fullon") != std::string::npos)
                data.UVbackground_redshift_fullon = std::stoi(value);
            if (key.find("UVbackground_redshift_drop") != std::string::npos)
                data.UVbackground_redshift_drop = std::stoi(value);
            if (key.find("UVbackground_redshift_off") != std::string::npos)
                data.UVbackground_redshift_off = std::stoi(value);
            if (key.find("Gamma") != std::string::npos) data.Gamma = std::stoi(value);
            if (key.find("three_body_rate") != std::string::npos) data.three_body_rate = std::stoi(value);
            if (key.find("cie_cooling") != std::string::npos) data.cie_cooling = std::stoi(value);
            if (key.find("h2_optical_depth_approximation") != std::string::npos)
                data.h2_optical_depth_approximation = std::stoi(value);
            if (key.find("photoelectric_heating_rate") != std::string::npos)
                data.photoelectric_heating_rate = std::stoi(value);
            if (key.find("Compton_xray_heating") != std::string::npos) data.Compton_xray_heating = std::stoi(value);
            if (key.find("LWbackground_intensity") != std::string::npos) data.LWbackground_intensity = std::stoi(value);
            if (key.find("LWbackground_sawtooth_suppression") != std::string::npos)
                data.LWbackground_sawtooth_suppression = std::stoi(value);
            if (key.find("use_volumetric_heating_rate") != std::string::npos)
                data.use_volumetric_heating_rate = std::stoi(value);
            if (key.find("use_specific_heating_rate") != std::string::npos)
                data.use_specific_heating_rate = std::stoi(value);
            if (key.find("use_radiative_transfer") != std::string::npos) data.use_radiative_transfer = std::stoi(value);
            if (key.find("radiative_transfer_coupled_rate_solver") != std::string::npos)
                data.radiative_transfer_coupled_rate_solver = std::stoi(value);
            if (key.find("radiative_transfer_intermediate_step") != std::string::npos)
                data.radiative_transfer_intermediate_step = std::stoi(value);
            if (key.find("radiative_transfer_hydrogen_only") != std::string::npos)
                data.radiative_transfer_hydrogen_only = std::stoi(value);
            if (key.find("H2_self_shielding") != std::string::npos) data.H2_self_shielding = std::stoi(value);
            if (key.find("dust_chemistry") != std::string::npos) data.dust_chemistry = std::stoi(value);
        };

        FILE *file = fopen(grackle_options_file_path.c_str(), "r");
        char key[32], value[64];

        while (fscanf(file, "%31s = %63s", key, value) == 2) {
            setGrackleOption(global_values.data, std::string(key), std::string(value));
        }
        std::cout << global_values.data.grackle_data_file << std::endl;
        fclose(file);
    }

//Implementation of cooling functions
    template<typename T>
    struct cooling_field_data_content {
        grackle_field_data data;
        int zero[3] = {0, 0, 0};
        int one[3] = {1, 1, 1};
        gr_float gr_rho;
        gr_float gr_u;
        gr_float x_velocity;
        gr_float y_velocity;
        gr_float z_velocity;
        gr_float HI_density;
        gr_float HII_density;
        gr_float HM_density;
        gr_float HeI_density;
        gr_float HeII_density;
        gr_float HeIII_density;
        gr_float H2I_density;
        gr_float H2II_density;
        gr_float DI_density;
        gr_float DII_density;
        gr_float HDI_density;
        gr_float e_density;
        gr_float metal_density;
        gr_float volumetric_heating_rate_gr;
        gr_float specific_heating_rate_gr;
        gr_float RT_heating_rate_gr;
        gr_float RT_HI_ionization_rate_gr;
        gr_float RT_HeI_ionization_rate_gr;
        gr_float RT_HeII_ionization_rate_gr;
        gr_float RT_H2_dissociation_rate_gr;
        gr_float H2_self_shielding_length_gr;


        void assign_field_data(T &rho, T &u, T &HI_fraction,
                               T &HII_fraction, T &HM_fraction, T &HeI_fraction, T &HeII_fraction, T &HeIII_fraction,
                               T &H2I_fraction, T &H2II_fraction, T &DI_fraction, T &DII_fraction, T &HDI_fraction,
                               T &e_fraction,
                               T &metal_fraction, T &volumetric_heating_rate, T &specific_heating_rate,
                               T &RT_heating_rate,
                               T &RT_HI_ionization_rate, T &RT_HeI_ionization_rate, T &RT_HeII_ionization_rate,
                               T &RT_H2_dissociation_rate, T &H2_self_shielding_length) {
            data.grid_rank = 3;
            data.grid_dimension = one;
            data.grid_start = zero;
            data.grid_end = zero;
            data.grid_dx = 0.0;

            gr_rho = (gr_float) rho;
            gr_u = (gr_float) u;
            x_velocity = 0.;
            y_velocity = 0.;
            z_velocity = 0.;
            HI_density = (gr_float) HI_fraction * (gr_float) rho;
            HII_density = (gr_float) HII_fraction * (gr_float) rho;
            HM_density = (gr_float) HM_fraction * (gr_float) rho;
            HeI_density = (gr_float) HeI_fraction * (gr_float) rho;
            HeII_density = (gr_float) HeII_fraction * (gr_float) rho;
            HeIII_density = (gr_float) HeIII_fraction * (gr_float) rho;
            H2I_density = (gr_float) H2I_fraction * (gr_float) rho;
            H2II_density = (gr_float) H2II_fraction * (gr_float) rho;
            DI_density = (gr_float) DI_fraction * (gr_float) rho;
            DII_density = (gr_float) DII_fraction * (gr_float) rho;
            HDI_density = (gr_float) HDI_fraction * (gr_float) rho;
            e_density = (gr_float) e_fraction * (gr_float) rho;
            metal_density = (gr_float) metal_fraction * (gr_float) rho;
            volumetric_heating_rate_gr = (gr_float) volumetric_heating_rate;
            specific_heating_rate_gr = (gr_float) specific_heating_rate;
            RT_heating_rate_gr = (gr_float) RT_heating_rate;
            RT_HI_ionization_rate_gr = (gr_float) RT_HI_ionization_rate;
            RT_HeI_ionization_rate_gr = (gr_float) RT_HeI_ionization_rate;
            RT_HeII_ionization_rate_gr = (gr_float) RT_HeII_ionization_rate;
            RT_H2_dissociation_rate_gr = (gr_float) RT_H2_dissociation_rate;
            H2_self_shielding_length_gr = (gr_float) H2_self_shielding_length;

            data.density = &gr_rho;
            data.internal_energy = &gr_u;
            data.x_velocity = &x_velocity;
            data.y_velocity = &y_velocity;
            data.z_velocity = &z_velocity;
            data.HI_density = &HI_density;
            data.HII_density = &HII_density;
            data.HeI_density = &HeI_density;
            data.HeII_density = &HeII_density;
            data.HeIII_density = &HeIII_density;
            data.e_density = &e_density;
            data.HM_density = &HM_density;
            data.H2I_density = &H2I_density;
            data.H2II_density = &H2II_density;
            data.DI_density = &DI_density;
            data.DII_density = &DII_density;
            data.HDI_density = &HDI_density;
            data.metal_density = &metal_density;

            data.volumetric_heating_rate = &volumetric_heating_rate_gr;
            data.specific_heating_rate = &specific_heating_rate_gr;
            data.RT_heating_rate = &RT_heating_rate_gr;
            data.RT_HI_ionization_rate = &RT_HI_ionization_rate_gr;
            data.RT_HeI_ionization_rate = &RT_HeI_ionization_rate_gr;
            data.RT_HeII_ionization_rate = &RT_HeII_ionization_rate_gr;
            data.RT_H2_dissociation_rate = &RT_H2_dissociation_rate_gr;
            data.H2_self_shielding_length = &H2_self_shielding_length_gr;
        };

        void get_field_data(T &rho, T &u, T &HI_fraction,
                            T &HII_fraction, T &HM_fraction, T &HeI_fraction, T &HeII_fraction, T &HeIII_fraction,
                            T &H2I_fraction, T &H2II_fraction, T &DI_fraction, T &DII_fraction, T &HDI_fraction,
                            T &e_fraction,
                            T &metal_fraction, T &volumetric_heating_rate, T &specific_heating_rate,
                            T &RT_heating_rate,
                            T &RT_HI_ionization_rate, T &RT_HeI_ionization_rate, T &RT_HeII_ionization_rate,
                            T &RT_H2_dissociation_rate, T &H2_self_shielding_length) {
            rho = gr_rho;
            u = gr_u;
            HI_fraction = HI_density / gr_rho;
            HII_fraction = HII_density / gr_rho;
            HM_fraction = HM_density / gr_rho;
            HeI_fraction = HeI_density / gr_rho;
            HeII_fraction = HeII_density / gr_rho;
            HeIII_fraction = HeIII_density / gr_rho;
            H2I_fraction = H2I_density / gr_rho;
            H2II_fraction = H2II_density / gr_rho;
            DI_fraction = DI_density / gr_rho;
            DII_fraction = DII_density / gr_rho;
            HDI_fraction = HDI_density / gr_rho;
            e_fraction = e_density / gr_rho;
            metal_fraction = metal_density / gr_rho;
            volumetric_heating_rate = volumetric_heating_rate_gr;
            specific_heating_rate = specific_heating_rate_gr;
            RT_heating_rate = RT_heating_rate_gr;
            RT_HI_ionization_rate = RT_HI_ionization_rate_gr;
            RT_HeI_ionization_rate = RT_HeI_ionization_rate_gr;
            RT_HeII_ionization_rate = RT_HeII_ionization_rate_gr;
            RT_H2_dissociation_rate = RT_H2_dissociation_rate_gr;
            H2_self_shielding_length = H2_self_shielding_length_gr;
        }
    };


    template<typename T>
    void cool_particle(GlobalValues &gv, const T &dt, T &rho, T &u, T &HI_fraction,
                       T &HII_fraction, T &HM_fraction, T &HeI_fraction, T &HeII_fraction, T &HeIII_fraction,
                       T &H2I_fraction, T &H2II_fraction, T &DI_fraction, T &DII_fraction, T &HDI_fraction,
                       T &e_fraction,
                       T &metal_fraction, T &volumetric_heating_rate, T &specific_heating_rate, T &RT_heating_rate,
                       T &RT_HI_ionization_rate, T &RT_HeI_ionization_rate, T &RT_HeII_ionization_rate,
                       T &RT_H2_dissociation_rate, T &H2_self_shielding_length) {
        cooling_field_data_content<T> grackle_fields;
        grackle_fields.assign_field_data(rho, u, HI_fraction,
                                         HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction,
                                         H2I_fraction, H2II_fraction, DI_fraction, DII_fraction, HDI_fraction,
                                         e_fraction,
                                         metal_fraction, volumetric_heating_rate, specific_heating_rate,
                                         RT_heating_rate,
                                         RT_HI_ionization_rate, RT_HeI_ionization_rate, RT_HeII_ionization_rate,
                                         RT_H2_dissociation_rate, H2_self_shielding_length);


        // Grackle uses 0 as a return code to indicate failure
        if (0 == local_solve_chemistry(&gv.data, &gv.rates, &gv.units, &grackle_fields.data, dt)) {
            throw std::runtime_error("Grackle: Error in local_solve_chemistry");
        }
        grackle_fields.get_field_data(rho, u, HI_fraction,
                                      HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction,
                                      H2I_fraction, H2II_fraction, DI_fraction, DII_fraction, HDI_fraction, e_fraction,
                                      metal_fraction, volumetric_heating_rate, specific_heating_rate, RT_heating_rate,
                                      RT_HI_ionization_rate, RT_HeI_ionization_rate, RT_HeII_ionization_rate,
                                      RT_H2_dissociation_rate, H2_self_shielding_length);

    }

    template void cool_particle<double>(GlobalValues &, const double &, double &, double &, double &,
                                        double &, double &, double &, double &, double &,
                                        double &, double &, double &, double &, double &,
                                        double &, double &, double &, double &, double &,
                                        double &, double &, double &, double &, double &);

    template void cool_particle<float>(GlobalValues &, const float &, float &, float &, float &,
                                       float &, float &, float &, float &, float &,
                                       float &, float &, float &, float &, float &,
                                       float &, float &, float &, float &, float &,
                                       float &, float &, float &, float &, float &);

    template<typename T>
    T energy_to_temperature(GlobalValues &gv, const T &dt, T &rho, T &u, T &HI_fraction,
                            T &HII_fraction, T &HM_fraction, T &HeI_fraction, T &HeII_fraction, T &HeIII_fraction,
                            T &H2I_fraction, T &H2II_fraction, T &DI_fraction, T &DII_fraction, T &HDI_fraction,
                            T &e_fraction, T &metal_fraction, T &volumetric_heating_rate, T &specific_heating_rate,
                            T &RT_heating_rate, T &RT_HI_ionization_rate, T &RT_HeI_ionization_rate,
                            T &RT_HeII_ionization_rate, T &RT_H2_dissociation_rate, T &H2_self_shielding_length) {
        grackle_field_data grackle_fields;
        grackle_fields.grid_rank = 3;
        int zero[] = {0, 0, 0};
        int one[] = {1, 1, 1};
        grackle_fields.grid_dimension = one;
        grackle_fields.grid_start = zero;
        grackle_fields.grid_end = zero;
        grackle_fields.grid_dx = 0.0;

        gr_float gr_rho = (gr_float) rho;
        grackle_fields.density = &gr_rho;
        gr_float gr_u = (gr_float) u;
        grackle_fields.internal_energy = &gr_u;
        gr_float x_velocity = 0.;
        grackle_fields.x_velocity = &x_velocity;
        gr_float y_velocity = 0.;
        grackle_fields.y_velocity = &y_velocity;
        gr_float z_velocity = 0.;
        grackle_fields.z_velocity = &z_velocity;
        gr_float HI_density = (gr_float) HI_fraction * (gr_float) rho;
        gr_float HII_density = (gr_float) HII_fraction * (gr_float) rho;
        gr_float HM_density = (gr_float) HM_fraction * (gr_float) rho;
        gr_float HeI_density = (gr_float) HeI_fraction * (gr_float) rho;
        gr_float HeII_density = (gr_float) HeII_fraction * (gr_float) rho;
        gr_float HeIII_density = (gr_float) HeIII_fraction * (gr_float) rho;
        gr_float H2I_density = (gr_float) H2I_fraction * (gr_float) rho;
        gr_float H2II_density = (gr_float) H2II_fraction * (gr_float) rho;
        gr_float DI_density = (gr_float) DI_fraction * (gr_float) rho;
        gr_float DII_density = (gr_float) DII_fraction * (gr_float) rho;
        gr_float HDI_density = (gr_float) HDI_fraction * (gr_float) rho;
        gr_float e_density = (gr_float) e_fraction * (gr_float) rho;
        gr_float metal_density = (gr_float) metal_fraction * (gr_float) rho;
        gr_float volumetric_heating_rate_gr = (gr_float) volumetric_heating_rate;
        gr_float specific_heating_rate_gr = (gr_float) specific_heating_rate;
        gr_float RT_heating_rate_gr = (gr_float) RT_heating_rate;
        gr_float RT_HI_ionization_rate_gr = (gr_float) RT_HI_ionization_rate;
        gr_float RT_HeI_ionization_rate_gr = (gr_float) RT_HeI_ionization_rate;
        gr_float RT_HeII_ionization_rate_gr = (gr_float) RT_HeII_ionization_rate;
        gr_float RT_H2_dissociation_rate_gr = (gr_float) RT_H2_dissociation_rate;
        gr_float H2_self_shielding_length_gr = (gr_float) H2_self_shielding_length;

        grackle_fields.HI_density = &HI_density;
        grackle_fields.HII_density = &HII_density;
        grackle_fields.HeI_density = &HeI_density;
        grackle_fields.HeII_density = &HeII_density;
        grackle_fields.HeIII_density = &HeIII_density;
        grackle_fields.e_density = &e_density;
        grackle_fields.HM_density = &HM_density;
        grackle_fields.H2I_density = &H2I_density;
        grackle_fields.H2II_density = &H2II_density;
        grackle_fields.DI_density = &DI_density;
        grackle_fields.DII_density = &DII_density;
        grackle_fields.HDI_density = &HDI_density;
        grackle_fields.metal_density = &metal_density;

        grackle_fields.volumetric_heating_rate = &volumetric_heating_rate_gr;
        grackle_fields.specific_heating_rate = &specific_heating_rate_gr;
        grackle_fields.RT_heating_rate = &RT_heating_rate_gr;
        grackle_fields.RT_HI_ionization_rate = &RT_HI_ionization_rate_gr;
        grackle_fields.RT_HeI_ionization_rate = &RT_HeI_ionization_rate_gr;
        grackle_fields.RT_HeII_ionization_rate = &RT_HeII_ionization_rate_gr;
        grackle_fields.RT_H2_dissociation_rate = &RT_H2_dissociation_rate_gr;
        grackle_fields.H2_self_shielding_length = &H2_self_shielding_length_gr;

        gr_float temp;

        if (0 == local_calculate_temperature(&gv.data, &gv.rates, &gv.units, &grackle_fields, &temp) == 0) {
            throw std::runtime_error("Grackle: Error in local_calculate_temperature");
        }
        return temp;
    }

    template double energy_to_temperature<double>(GlobalValues &, const double &, double &, double &, double &,
                                                  double &, double &, double &, double &, double &,
                                                  double &, double &, double &, double &, double &,
                                                  double &, double &, double &, double &, double &,
                                                  double &, double &, double &, double &, double &);

    template float energy_to_temperature<float>(GlobalValues &, const float &, float &, float &, float &,
                                                float &, float &, float &, float &, float &,
                                                float &, float &, float &, float &, float &,
                                                float &, float &, float &, float &, float &,
                                                float &, float &, float &, float &, float &);

}