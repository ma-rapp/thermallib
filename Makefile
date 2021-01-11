all: HotSpot-6.0/.stamp_download
	make -C MatEx all
	make -C HotSpot-6.0 all

HotSpot-6.0/.stamp_download: HotSpot-6.0.zip
	unzip $<
	mv uvahotspot-HotSpot-* HotSpot-6.0
	touch $@

HotSpot-6.0.zip:
	wget https://github.com/uvahotspot/hotspot/zipball/master/ -O HotSpot-6.0.zip

clean:
	make -C MatEx clean
	rm -rf HotSpot-6.0
	rm -f HotSpot-6.0.zip
