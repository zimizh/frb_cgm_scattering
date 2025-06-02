
import numpy as np
import utilities as ut

# default FIRE model for stellar evolution to assume throughout
FIRE_MODEL_DEFAULT = 'fire2'

class read_gizmo():
    def __init__():
        return
    
    def prop(self, property_name, indices=None, _dict_only=False):
        '''
        Get property, either stored in self's dictionary or derive it from stored properties.
        Can compute basic mathematical manipulations/combinations, for example:
            'log temperature', 'temperature / density', 'abs position'

        Parameters
        ----------
        property_name : str
            name of property
        indices : array
            indices of particles to get properties of
        _dict_only : bool
            require property_name to be in self's dict - avoids endless recursion
            primarily for internal/recursive usage of this function

        Returns
        -------
        values : float or array
            depending on dimensionality of input indices
        '''
        # parsing general to all catalogs ----------
        property_name = property_name.strip()  # strip white space

        # if input is in self dictionary, return as is
        if property_name in self:
            if indices is not None:
                return self[property_name][indices]
            else:
                return self[property_name]
        elif _dict_only:
            raise KeyError(f'property = {property_name} is not in self\'s dictionary')

        # math relation, combining more than one property
        if (
            '/' in property_name
            or '*' in property_name
            or '+' in property_name
            or '-' in property_name
        ):
            prop_names = property_name

            for delimiter in ['/', '*', '+', '-']:
                if delimiter in property_name:
                    prop_names = prop_names.split(delimiter)
                    break

            if len(prop_names) == 1:
                raise KeyError(f'not sure how to parse property = {property_name}')

            # make copy so not change values in input catalog
            prop_values = np.array(self.prop(prop_names[0], indices))

            for prop_name in prop_names[1:]:
                if '/' in property_name:
                    if np.isscalar(prop_values):
                        if self.prop(prop_name, indices) == 0:
                            prop_values = np.nan
                        else:
                            prop_values = prop_values / self.prop(prop_name, indices)
                    else:
                        masks = self.prop(prop_name, indices) != 0
                        prop_values[masks] = (
                            prop_values[masks] / self.prop(prop_name, indices)[masks]
                        )
                        masks = self.prop(prop_name, indices) == 0
                        prop_values[masks] = np.nan
                if '*' in property_name:
                    prop_values = prop_values * self.prop(prop_name, indices)
                if '+' in property_name:
                    prop_values = prop_values + self.prop(prop_name, indices)
                if '-' in property_name:
                    prop_values = prop_values - self.prop(prop_name, indices)

            if prop_values.size == 1:
                prop_values = np.float64(prop_values)

            return prop_values

        # math transformation of single property
        if property_name[:3] == 'log':
            return ut.math.get_log(self.prop(property_name.replace('log', ''), indices))

        if property_name[:3] == 'abs':
            return np.abs(self.prop(property_name.replace('abs', ''), indices))

        # parsing specific to this catalog ----------
        # stellar mass loss
        if ('mass' in property_name and 'form' in property_name) or 'mass.loss' in property_name:
            if (
                'fire.model' in self.info
                and isinstance(self.info['fire.model'], str)
                and len(self.info['fire.model']) > 0
            ):
                fire_model = self.info['fire.model']
            else:
                fire_model = FIRE_MODEL_DEFAULT

            if 'MassLoss' not in self.__dict__ or self.MassLoss is None:
                from . import gizmo_star

                # create class to compute/store stellar mass loss as a function of age, metallicity
                self.MassLoss = gizmo_star.MassLossClass(model=fire_model)

            # fractional mass loss since formation
            if 'fire2' in fire_model:
                metal_mass_fractions = self.prop('massfraction.metals', indices)
            elif 'fire3' in fire_model:
                metal_mass_fractions = self.prop('massfraction.iron', indices)
            values = self.MassLoss.get_mass_loss_from_spline(
                self.prop('age', indices) * 1000,
                metal_mass_fractions=metal_mass_fractions,
            )

            if 'mass.loss' in property_name:
                if 'fraction' in property_name:
                    pass
                else:
                    values *= self.prop('mass', indices, _dict_only=True) / (
                        1 - values
                    )  # mass loss
            elif 'mass' in property_name and 'form' in property_name:
                values = self.prop('mass', indices, _dict_only=True) / (
                    1 - values
                )  # formation mass

            return values

        # mass of single element
        if 'mass.' in property_name:
            # mass from individual element
            values = self.prop('mass', indices, _dict_only=True) * self.prop(
                property_name.replace('mass.', 'massfraction.'), indices
            )

            if property_name == 'mass.hydrogen.neutral':
                # mass from neutral hydrogen (excluding helium, metals, and ionized hydrogen)
                values = values * self.prop('hydrogen.neutral.fraction', indices, _dict_only=True)

            elif property_name == 'mass.hydrogen.ionized':
                # mass from neutral hydrogen (excluding helium, metals, and ionized hydrogen)
                values = values * (
                    1 - self.prop('hydrogen.neutral.fraction', indices, _dict_only=True)
                )

            return values

        # elemental abundance
        if 'massfraction' in property_name or 'metallicity' in property_name:
            return self._get_abundances(property_name, indices)

        if 'number.density' in property_name:
            values = (
                self.prop('density', indices, _dict_only=True)
                * ut.constant.proton_per_sun
                * ut.constant.kpc_per_cm**3
            )

            if 'hydrogen' in property_name:
                # number density of hydrogen, using actual hydrogen mass of each particle [cm ^ -3]
                values = values * self.prop('massfraction.hydrogen', indices)
            else:
                # number density of 'hydrogen', assuming solar metallicity for particles [cm ^ -3]
                values = values * ut.constant.sun_massfraction['hydrogen']

            return values

        if 'size' in property_name:
            # default size := inter-particle spacing = (mass / density)^(1/3) [kpc]
            f = (np.pi / 3) ** (1 / 3) / 2  # 0.5077, converts from default size to full extent

            if 'size' in self:
                values = self.prop('size', indices, _dict_only=True)
            else:
                values = (
                    self.prop('mass', indices, _dict_only=True)
                    / self.prop('density', indices, _dict_only=True)
                ) ** (1 / 3)

            if 'plummer' in property_name:
                # convert to plummer equivalent
                values = values / f / 2.8
            elif 'max' in property_name:
                # convert to maximum extent of kernel (radius of compact support)
                values = values / f

            if '.pc' in property_name:
                # convert to [pc]
                values = values * 1000

            return values

        if 'volume' in property_name:
            # volume := mass / density [kpc^3]
            if 'size' in self:
                return self.prop('size', indices, _dict_only=True) ** 3
            else:
                return self.prop('mass', indices, _dict_only=True) / self.prop(
                    'density', indices, _dict_only=True
                )

            if '.pc' in property_name:
                # convert to [pc^3]
                values = values * 1e9

        # free-fall time of gas cell := (3 pi / (32 G rho)) ^ 0.5
        if 'time.freefall' in property_name:
            values = self.prop('density', indices, _dict_only=True)  # [M_sun / kpc ^ 3]
            values = (3 * np.pi / (32 * ut.constant.grav_kpc_msun_Gyr * values)) ** 0.5  # [Gyr]

            return values

        # hydrogen ionized fraction
        if property_name == 'hydrogen.ionized.fraction':
            return 1 - self.prop('hydrogen.neutral.fraction', indices, _dict_only=True)

        if 'magnetic' in property_name and (
            'energy' in property_name or 'pressure' in property_name
        ):
            # magnetic field: energy density = pressure = B^2 / (8 pi)
            # convert from stored [Gauss] to [erg / cm^3]
            values = self.prop('magnetic.field', indices, _dict_only=True)
            values = np.sum(values**2, 1) / (8 * np.pi)

            if 'energy' in property_name and 'density' not in property_name:
                # total energy in magnetic field [erg]
                values = values * self.prop('volume', indices) * ut.constant.cm_per_kpc**3

            return values

        if 'cosmicray.energy.density' in property_name:
            # energy density in cosmic rays [M_sun / kpc / Gyr^2]
            return self.prop('cosmicray.energy', indices, _dict_only=True) / (
                self.prop('volume', indices)
            )

        if 'photon.energy.density' in property_name:
            return self.prop('cosmicray.energy', indices, _dict_only=True) / (
                self.prop('volume', indices)
            )

        # mean molecular mass [g] or mass ratio [dimensionless]
        if 'molecular.mass' in property_name:
            helium_mass_fracs = self.prop('massfraction.helium', indices)
            ys_helium = helium_mass_fracs / (4 * (1 - helium_mass_fracs))
            molecular_mass_ratios = (1 + 4 * ys_helium) / (
                1 + ys_helium + self.prop('electron.fraction', indices, _dict_only=True)
            )
            values = molecular_mass_ratios

            if property_name == 'molecular.mass.ratio':
                pass
            elif property_name == 'molecular.mass':
                values *= ut.constant.proton_mass

            return values

        # internal energy of gas [cm^2 / s^2] - undo conversion to temperature
        if 'internal.energy' in property_name:
            molecular_masses = self.prop('molecular.mass', indices, _dict_only=True)

            values = self.prop('temperature') / (
                ut.constant.centi_per_kilo**2
                * (self.adiabatic_index - 1)
                * molecular_masses
                / ut.constant.boltzmann
            )

            return values

        # sound speed [km/s], for simulations that do not store it
        if 'sound.speed' in property_name:
            values = (
                np.sqrt(
                    self.adiabatic_index
                    * ut.constant.boltzmann
                    * self.prop('temperature', indices, _dict_only=True)
                    / ut.constant.proton_mass
                )
                * ut.constant.kilo_per_centi
            )

            return values

        # formation time [Gyr] or coordinates
        if (
            ('form.' in property_name or property_name == 'age')
            and 'host' not in property_name
            and 'distance' not in property_name
            and 'velocity' not in property_name
        ):
            if property_name == 'age' or ('time' in property_name and 'lookback' in property_name):
                # look-back time (stellar age) to formation
                values = self.snapshot['time'] - self.prop('form.time', indices)
            elif 'time' in property_name:
                # time (age of universe) of formation
                values = self.Cosmology.get_time(
                    self.prop('form.scalefactor', indices, _dict_only=True), 'scalefactor'
                )
            elif 'redshift' in property_name:
                # redshift of formation
                values = 1 / self.prop('form.scalefactor', indices, _dict_only=True) - 1
            elif 'snapshot' in property_name:
                # snapshot index immediately after formation
                # increase formation scale-factor slightly for safety, because scale-factors of
                # written snapshots do not exactly coincide with input scale-factors
                padding_factor = 1 + 1e-7
                values = self.Snapshot.get_snapshot_indices(
                    'scalefactor',
                    np.clip(
                        self.prop('form.scalefactor', indices, _dict_only=True) * padding_factor,
                        0,
                        1,
                    ),
                    round_kind='up',
                )

            return values

        # distance or velocity wrt the host galaxy/halo
        if 'host' in property_name and (
            'distance' in property_name
            or 'velocity' in property_name
            or 'acceleration' in property_name
        ):
            if 'host.' in property_name or 'host1.' in property_name:
                host_name = 'host.'
                host_index = 0
            elif 'host2.' in property_name:
                host_name = 'host2.'
                host_index = 1
            elif 'host3.' in property_name:
                host_name = 'host3.'
                host_index = 2
            else:
                raise ValueError(f'cannot identify host name in {property_name}')

            if 'form.' in property_name:
                # special case: coordinates wrt primary host *at formation*
                if 'distance' in property_name:
                    # 3-D distance vector wrt primary host at formation
                    values = self.prop('form.' + host_name + 'distance', indices, _dict_only=True)
                elif 'velocity' in property_name:
                    # 3-D velocity vectory wrt host at formation
                    values = self.prop('form.' + host_name + 'velocity', indices, _dict_only=True)
            else:
                # general case: coordinates wrt primary host at current snapshot
                if 'distance' in property_name:
                    # 3-D distance vector wrt the primary host
                    values = ut.coordinate.get_distances(
                        self.prop('position', indices, _dict_only=True),
                        self.host['position'][host_index],
                        self.info['box.length'],
                        self.snapshot['scalefactor'],
                    )  # [kpc physical]
                elif 'velocity' in property_name:
                    # 3-D velocity vector wrt the primary host, adding the Hubble flow
                    values = ut.coordinate.get_velocity_differences(
                        self.prop('velocity', indices, _dict_only=True),
                        self.host['velocity'][host_index],
                        self.prop('position', indices, _dict_only=True),
                        self.host['position'][host_index],
                        self.info['box.length'],
                        self.snapshot['scalefactor'],
                        self.snapshot['time.hubble'],
                    )
                elif 'acceleration' in property_name:
                    # 3-D acceleration
                    # no correction for Hubble flow
                    values = self.prop('acceleration', indices, _dict_only=True)
                    if 'acceleration' in self.host and len(self.host['acceleration']) > 0:
                        values -= self.host['acceleration']

                if 'principal' in property_name:
                    # align with host principal axes
                    assert (
                        len(self.host['rotation']) > 0
                    ), 'must assign hosts principal axes rotation tensor!'
                    values = ut.coordinate.get_coordinates_rotated(
                        values, self.host['rotation'][host_index]
                    )

            if '.cyl' in property_name or '.spher' in property_name:
                # convert to cylindrical or spherical coordinates
                if '.cyl' in property_name:
                    coordinate_system = 'cylindrical'
                elif '.spher' in property_name:
                    coordinate_system = 'spherical'

                if 'distance' in property_name:
                    values = ut.coordinate.get_positions_in_coordinate_system(
                        values, 'cartesian', coordinate_system
                    )
                elif 'velocity' in property_name or 'acceleration' in property_name:
                    if 'form.' in property_name:
                        # special case: coordinates wrt primary host *at formation*
                        distance_vectors = self.prop(
                            'form.' + host_name + 'distance', indices, _dict_only=True
                        )
                    elif 'principal' in property_name:
                        distance_vectors = self.prop(host_name + 'distance.principal', indices)
                    else:
                        distance_vectors = self.prop(host_name + 'distance', indices)
                    values = ut.coordinate.get_velocities_in_coordinate_system(
                        values, distance_vectors, 'cartesian', coordinate_system
                    )

                if '.rad' in property_name:
                    values = values[:, 0]
                elif '.azi' in property_name:
                    values = values[:, 1]
                elif '.ver' in property_name:
                    values = values[:, 2]

            # compute total (scalar) of distance
            if '.total' in property_name:
                if len(values.shape) == 1:
                    shape_pos = 0
                else:
                    shape_pos = 1
                values = np.sqrt(np.sum(values**2, shape_pos))

            return values

        # compute total (scalar) value from 3-D for some other property
        # such as velocity, acceleration, magnetic field
        if '.total' in property_name:
            prop_name = property_name.replace('.total', '')
            try:
                values = self.prop(prop_name, indices)
                values = np.sqrt(np.sum(values**2, 1))
                return values
            except ValueError:
                pass

        # should not get this far without a return
        raise KeyError(f'not sure how to parse property = {property_name}')