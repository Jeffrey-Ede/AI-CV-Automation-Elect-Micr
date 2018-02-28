//Initialise the electron microscope for interfacing
string buffer_loc = "//flexo.ads.warwick.ac.uk/shared39/EOL2100/2100/Users/Jeffrey-Ede/buffer/";
string change_filename = buffer_loc + "X.txt"; //Indicator file. Its existance indicates that instructions have been issued
string instr_filename = buffer_loc + "instr.txt"; //Python interface instructions
string state_filename = buffer_loc + "state.txt"; //Output microscope state information

number state_change_wait = 0.01 //Time delay between checks for new instructions in ms

//Global variables
 number Camb=2672, Camr=2688//Global values, for Orius SC600 camera
 object img_src

//Function UpdateCameraImage
//Gets next available frame from the camera 
void UpdateCameraImage(object img_src, image img)
{
  // Wait for next frame
  number acq_params_changed=0;
  number max_wait = 0.1;
  if (!img_src.IMGSRC_AcquireTo(img,true,max_wait,acq_params_changed))
  {   
	while (!img_src.IMGSRC_AcquireTo(img,false,max_wait,acq_params_changed))
	{
	}
  }	
}

// Stop any current camera viewer
number close_view=1, stop_view=0;
try
{
  cm_stopcurrentcameraViewer(stop_view);
}
catch
{
  throw("Couldn't stop camera properly, try again!");
}

//Works in imaging mode
 EMChangeMode("MAG1")
 external sem = NewSemaphore();
 try
 {
   ModelessDialog("Please spread the beam, large condenser aperture, feature in the centre","OK",sem)
   GrabSemaphore(sem)
   ReleaseSemaphore(sem)
   FreeSemaphore(sem)
 }
 catch
 {
   FreeSemaphore(sem)
   break;
 }

//Start the camera running in fast mode
//Use current camera
object camera = CMGetCurrentCamera();
// Create standard parameters
number data_type=2;
number kUnprocessed = 1;
number kDarkCorrected = 2;
number kGainNormalized = 3;
number processing = kGainNormalized;	
// Define camera parameter set
object acq_params = camera.CM_CreateAcquisitionParameters_FullCCD(processing,expo,binning,binning);
acq_params.CM_SetDoContinuousReadout(true);
acq_params.CM_SetQualityLevel(0);//What this does is unclear
object acquisition = camera.CM_CreateAcquisition(acq_params);
object frame_set_info = acquisition.CM_ACQ_GetDetector().DTCTR_CreateFrameSetInfo();
img_src = alloc(CM_AcquisitionImageSource).IMGSRC_Init(acquisition,frame_set_info,0);
CM_ClearDarkImages()//May not be necessary

img1:=acquisition.CM_CreateImageForAcquire( "Live" );
img1.DisplayAt(10,30);

//Asynchroneously lauch the neural network and python interface
string process = "python em_env.py "+change_file+" "+instr_file+" "+state_file+" "+state_change_wait;
LaunchExternalProcessAsync( process );

//Enthral the microscope to the interfacing files
number marionette = 1;
while ( marionette )
{
	//Wait for the interface to send instructions
	while ( !DoesFileExist( change_filename ) )
	{
		sleep( state_change_wait ); //Wait before checking again
	}
		
	//Prepare buffering filehandles
	number instr_file = OpenFileForReading( instr_filename );
	number state_file = OpenFileForWriting( state_filename );

	//Sequentially execute instructions issued by the python em_env interface
	string line;
	while ( ReadFileLine( instr_file, 0, *line ) )
	{
		//Determine the instruction
		number instr_num = val(line);
		number read_success;
		
		if( instr_num == 0 )
		{
			/*get_img*/
			image img;
			img_src.IMGSRC_BeginAcquisition()
			UpdateCameraImage(img_src,img);
			
			//Save the image
			read_success = ReadFileLine( instr_file, 0, *line );
			SaveImage( img, line );
			
			//Update the state file
			WriteFile( state_file, "0,"+line+"\n" );
		}
		if( instr_num == 1 )
		{
			/*EMSetStageX*/
			read_success = ReadFileLine( instr_file, 0, *line );
			number x_shift = val(line);
			
			//Get the curret z position of the stage
			number x = EMGetStageX();
			
			//Shift the stage
			EMSetStageX( x + x_shift );
		}
		if( instr_num == 2 )
		{
			/*EMSetStageY*/
			read_success = ReadFileLine( instr_file, 0, *line );
			number y_shift = val(line);
			
			//Get the curret z position of the stage
			number y = EMGetStageY();
			
			//Shift the stage
			EMSetStageY( y + y_shift );
			
		}
		if( instr_num == 3 )
		{
			/*Shift z*/
			read_success = ReadFileLine( instr_file, 0, *line );
			number z_shift = val(line);
			
			//Get the curret z position of the stage
			number z = EMGetStageZ();
			
			//Shift the stage
			EMSetStageZ( z + z_shift );
		}
		if( instr_num == 4 )
		{
			/*EMChangeBeamShift*/
			read_success = ReadFileLine( instr_file, 0, *line );
			number x_shift = val(line);
			read_success = ReadFileLine( instr_file, 0, *line );
			number y_shift = val(line);

			
			//Shift the beam
			EMChangeBeamShift( x_shift, y_shift )
		}
		if( instr_num == 5 )
		{
			/*terminate*/
			marionette = 0;
		}
	}

	//Release filehandles so that the files can be used by the python interface
	CloseFile( instr_file );
	CloseFile( state_file );
	
	//Destroy the interfacing indicator file to succeed control to the python interface
	DeleteFile( change_file );
}