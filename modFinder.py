from modulefinder import ModuleFinder
finder = ModuleFinder()

finder.run_script('app.py')

for module in finder.modules.keys():
    print(module)