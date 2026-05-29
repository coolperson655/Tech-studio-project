# import pygame
# CALIBRATION_RED = pygame.Color(252, 65, 3)
# CALIBRATION_YELLOW = pygame.Color(252, 219, 3)
# check_color_dict = {'IM_COLOR':['Waiting...',CALIBRATION_RED], 'RP_COLOR': ['Waiting...',CALIBRATION_RED], 'T_COLOR': ['Waiting...',CALIBRATION_RED]}
# print(check_color_dict["IM_COLOR"][1], check_color_dict["RP_COLOR"][1], check_color_dict["T_COLOR"][1])
# pygame.init()
# clock = pygame.time.Clock()
# while True:
#     pygame.display.set_caption("Calibration Test")

#     screen = pygame.display.set_mode((900, 600))
#     screen.fill(CALIBRATION_RED)
#     # pygame.draw.rect(screen, check_color_dict["IM_COLOR"][1], (40, 30, 440, 580), border_radius=10)
#     clock.tick(60)
testlist = [1,2,3]
print(testlist[0])
testlist[0] = 5
print(testlist[0])

        message = font_body.render("Checking sensor data stream for calibration readiness...", True, TEXT_WHITE)
        try:
            curls = frame_queue.get(timeout=0.5).curl
            curl_rates = abs(np.array(curls) - np.array(previous_curl))
            previous_curl = curls
            if np.all(curl_rates <= MAX_CURL_RATE):
                curl_in_range = np.all(np.abs(np.array(curls) - TARGET_CURL) <=CURL_TOLERANCE)
                for i, color in zip(range(3), check_color_dict.keys()):
                    curl_diff = curls[i] - TARGET_CURL
                    if curl_diff <= CURL_TOLERANCE and curl_diff >= -1* CURL_TOLERANCE:
                        check_color_dict[color][1] = INDICATOR_GREEN
                        check_color_dict[color][0] = 'HOLD'
                    elif curl_diff > CURL_TOLERANCE * 2:
                        check_color_dict[color][1] = CALIBRATION_RED
                        check_color_dict[color][0] = 'CURL MUCH LESS'
                    elif curl_diff > CURL_TOLERANCE:
                        check_color_dict[color][1] = CALIBRATION_YELLOW
                        check_color_dict[color][0] = 'CURL LESS'
                    elif curl_diff > -1* CURL_TOLERANCE * 2:
                        check_color_dict[color][1] = CALIBRATION_RED
                        check_color_dict[color][0] = 'CURL MUCH MORE'
                    elif curl_diff > -1* CURL_TOLERANCE:
                        check_color_dict[color][1] = CALIBRATION_YELLOW
                        check_color_dict[color][0] = 'CURL MORE'
                if curl_in_range:
                    message = font_body.render("Calibration successful! Tracking stable and within target range. Proceeding to calibration interface...", True, TEXT_WHITE)
                    APP_STATE = "CALIBRATION"
            else:
                message = font_body.render("'tracking unstable please hold still and get in view of the camera", True, TEXT_WHITE)
                check_color_dict = {'IM_COLOR':['Waiting...',CALIBRATION_RED], 'RP_COLOR': ['Waiting...',CALIBRATION_RED], 'T_COLOR': ['Waiting...',CALIBRATION_RED]}
        except:
            message = font_body.render("No data stream detected. Please ensure sensors are active and hand is in view.", True, TEXT_WHITE)
        # Dict keys | [text, color]
        # IM_COLOR 
        # RP_COLOR  
        # T_COLOR
        screen.blit(message, (60, 100))
        screen.blit(font_small.render("Target Curl: {:.2f} | Tolerance: ±{:.2f} | Max Rate: {:.2f}".format(TARGET_CURL, CURL_TOLERANCE, MAX_CURL_RATE), True, TEXT_MUTED), (60, 130))
        screen.blit(font_small.render(f"Curl Rates: {np.max(curl_rates) if 'curl_rates' in locals() else 0.0}, {curls[0] if 'curls' in locals() else 0.0}, {curls[1] if 'curls' in locals() else 0.0}, {curls[2] if 'curls' in locals() else 0.0}", True, TEXT_MUTED), (60, 150))
        screen.blit(font_small.render("Instructions: Hold each finger steady at the target curl position. Green indicates ready, yellow indicates adjust curl, red indicates out of range.", True, TEXT_MUTED), (60, 170))
        screen.blit(font_small.render('IM Curl:', True, TEXT_WHITE), (60, 190))
        screen.blit(font_small.render('RP Curl:', True, TEXT_WHITE), (360, 190))
        screen.blit(font_small.render('T Curl:', True, TEXT_WHITE), (650, 190))
        left = draw_ui_button(screen, (50, 220, 260, 160), check_color_dict["IM_COLOR"][0],check_color_dict['IM_COLOR'][1],TEXT_BLACK)
        middle =draw_ui_button(screen, (350, 220, 260, 160), check_color_dict["RP_COLOR"][0], check_color_dict['RP_COLOR'][1],TEXT_BLACK)
        right = draw_ui_button(screen, (650, 220, 260, 160), check_color_dict["T_COLOR"][0], check_color_dict['T_COLOR'][1],TEXT_BLACK)
        time.sleep(0.1) # Small delay to prevent excessive CPU usage during checking loop