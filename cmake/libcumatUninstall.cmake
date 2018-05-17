# Uninstalls libcumat files mentioned in install_manifest.txt from the install folder

set(MANIFEST ${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt)

if(EXISTS ${MANIFEST})
	message(STATUS "========= Uninstalling libcumat =========")

	file(STRINGS ${MANIFEST} files)

	foreach(file ${files})
		if(EXISTS ${file})
			message(STATUS "Removing file: '${file}'")

			execute_process(
				COMMAND ${CMAKE_COMMAND} -E remove ${file}
				OUTPUT_VARIABLE rm_out
				RESULT_VARIABLE rm_res
			)

			if(NOT ${rm_res} STREQUAL 0)
				message(FATAL_ERROR "Failed to remove file: '${file}'.")
			endif()
		else()
			message(STATUS "File '${file}' not found.")
		endif()
	endforeach(file)

	message(STATUS "========= Finished uninstalling libcumat =========")
else()
	message(STATUS "Cannot find install manifest: '${MANIFEST}'")
	message(STATUS "Make sure library is installed by running \"make install\" first")
	message(STATUS "and check that install_manifest.txt exists the build folder.")
endif()
