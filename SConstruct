import os;
# list of all sconscript files
# ORDER-SENSITIVE!
# add your sconscript only AFTER all dependent
all_ss = [
		'utils/SConscript',
		'alg/SConscript',
		'ga_client/SConscript',
		'gis_neuro/SConscript'
		]

# process custom settings
custom_env = Environment(ENV = os.environ);
# use distributed compilation
#custom_env['CC'] = ['distcc'];
#custom_env['CXX'] = ['distcc'];
custom_env.Append(
	CCFLAGS = ['-W', '-Wall', '-Wno-deprecated', '-Werror=return-type', '-pthread'], #'-fvisibility=hidden', '-fvisibility-inlines-hidden'],
	CPPPATH = ['/home/uentity/lib/tbb/include', '/home/uentity/lib/boost', '#utils', '#utils/include', '#alg/include'],
	LIBPATH = ['/home/uentity/lib/boost/lib']
);

#Help(cvars.GenerateHelpText(custom_env));
Export('custom_env');

vars = Variables();
vars.Add('debug', 'Set to 1 to build debug version of hybrid_adapt libs', '1');
vars.Add('release', 'Set to 1 to build release version of hybrid_adapt libs', '0');
vars.Update(custom_env);
# try to change def value of build variable
#custom_env.Replace(debug = 0);

# generate variables help
Help(vars.GenerateHelpText(custom_env));

#print('debug option is: ', custom_env['debug']);
#print('subst debug option is: ', custom_env.subst('${debug}'));
#print('release option is: ', custom_env['release']);
#print('subst release option is: ', custom_env.subst('${release}'));

# Add options telling whether to build release and debug
# build debug by default
#AddOption('--ha_debug', dest='ha_debug', help='Set to 1 for building debug', type = 'int', default=1);
#AddOption('--ha_release', dest='ha_release', help='Set to 1 for building release', type = 'int', default=0);

# parse scons files
[SConscript(x) for x in all_ss];

