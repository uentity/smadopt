import os;
import os.path as osp;

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

vars = Variables();
vars.Add('debug', 'Set to 1 to build debug version of hybrid_adapt libs', '1');
vars.Add('release', 'Set to 1 to build release version of hybrid_adapt libs', '0');
vars.Add('bs', 'Set to 1 to build BlueSky compatibility layer', '1');
vars.Add('python_name', 'Specify actual Python interpreter name (with version)', 'python2.7');
vars.Update(custom_env);
# try to change def value of build variable
#custom_env.Replace(debug = 0);

# use distributed compilation
#custom_env['CC'] = ['distcc'];
#custom_env['CXX'] = ['distcc'];
custom_env.Append(
	CCFLAGS = ['-W', '-Wall', '-Wno-deprecated', '-Werror=return-type', '-pthread'], #'-fvisibility=hidden', '-fvisibility-inlines-hidden'],
	CPPPATH = [[os.environ['BOOST_PATH']], '#utils', '#utils/include', '#alg/include', '/usr/include/${python_name}'],
	LIBPATH = ['/home/uentity/lib/boost/lib'],
	RPATH = ['/home/uentity/lib/boost/lib'],
	CPPDEFINES = ['UNIX', 'PYTHON_VERSION=27']
);

if custom_env["bs"] == '1' :
	custom_env.Append(
		CPPDEFINES = [
			'BLUE_SKY_COMPAT', 'BS_EXPORTING_PLUGIN', 'BSPY_EXPORTING_PLUGIN',
			'PYTHON_VERSION=27', 'BS_EXCEPTION_USE_BOOST_FORMAT', 'BS_DISABLE_MT_LOCKS'
		],
		CPPPATH = [
			osp.join(os.environ['BLUE_SKY_PATH'], 'kernel', 'include'),
			osp.join(os.environ['BLUE_SKY_PATH'], 'kernel', 'include', 'python'),
			osp.join(os.environ['BLUE_SKY_PATH'], 'plugins', 'bs-eagle', 'bs_bos_core_base', 'include')
		],
		LIBPATH = [
			osp.join(os.environ['BLUE_SKY_PATH'], 'exe', 'debug'),
			osp.join(os.environ['BLUE_SKY_PATH'], 'exe', 'release')
		],
	);

Export('custom_env');
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

