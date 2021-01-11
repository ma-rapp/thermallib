all: MatEx

MatEx: .stamp_matex_source .stamp_eigen_source
	g++ -Ieigen-3.3.9  matex.cpp configurationParameters.cpp floorplan.cpp rcModel.cpp -o MatEx

.stamp_eigen_source: eigen-3.3.9.zip
	unzip $<
	touch $@

eigen-3.3.9.zip:
	wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip

.stamp_matex_source: MatEx-1.0.zip
	unzip $<
	touch $@

MatEx-1.0.zip:
	wget http://cesdownloads.itec.kit.edu/download/download.php?id=MatEx-1.0.zip -O $@

clean:
	rm -f MatEx *.zip *.cpp *.h *.bin LICENSE README *.flp *.ptrace *.config .stamp*
	rm -rf eigen-3.3.9
